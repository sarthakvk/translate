"""
Async orchestrator: VAD phrase → ASR → preprocess → translate → TTS.

Each WebSocket connection owns one Pipeline instance. The pipeline exposes:
  - feed(pcm_bytes) → async generator yielding WS messages
  - stop()          → flushes any buffered speech and yields final messages

When the user speaks while TTS is playing, the client pauses playback on
speech_start, then either resumes on speech_rejected or discards old playback
on speech_confirmed.

WS message shapes (dicts, serialised to JSON by main.py):
  {"type": "speech_start"}
  {"type": "speech_confirmed"}
  {"type": "speech_rejected"}
  {"type": "transcript", "en": str, "stage": "final"}
  {"type": "transcript", "en": str, "hi": str, "stage": "final"}
  {"type": "audio",      "data": <base64 MP3 str>}
  {"type": "error",      "message": str}
"""

import base64
import logging
import re
from typing import AsyncIterator

import numpy as np

from .config import (
    ASR_HALLUCINATION_PHRASES,
    ASR_HALLUCINATION_RMS,
    ASR_MIN_PEAK,
    ASR_MIN_RMS,
)
from .vad import VADProcessor
from .asr import transcribe
from .preprocessor import preprocess, postprocess
from .translator import translate
from .tts import synthesize

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self) -> None:
        self._vad = VADProcessor()
        self._prev_hindi = ""

    async def feed(self, pcm_bytes: bytes) -> AsyncIterator[dict]:
        event = self._vad.feed(pcm_bytes)
        if event.speech_started:
            yield {"type": "speech_start"}
        if event.phrase is not None:
            async for msg in self._run(event.phrase):
                yield msg

    async def stop(self) -> AsyncIterator[dict]:
        phrase = self._vad.flush_pending()
        if phrase is not None:
            async for msg in self._run(phrase):
                yield msg

    async def _run(self, samples: np.ndarray) -> AsyncIterator[dict]:
        try:
            rms, peak = _audio_energy(samples)
            if rms < ASR_MIN_RMS and peak < ASR_MIN_PEAK:
                logger.info("Dropping low-energy phrase before ASR: rms=%.4f peak=%.4f", rms, peak)
                yield {"type": "speech_rejected"}
                return

            english = await transcribe(samples)
            if not english:
                yield {"type": "speech_rejected"}
                return

            if _is_low_energy_hallucination(english, rms):
                logger.info("Dropping likely ASR hallucination %r: rms=%.4f", english, rms)
                yield {"type": "speech_rejected"}
                return

            yield {"type": "speech_confirmed"}
            yield {"type": "transcript", "en": english, "stage": "final"}

            processed = preprocess(english)
            if not processed.cleaned:
                return

            raw_hindi = await translate(processed, self._prev_hindi)
            if not raw_hindi:
                return

            hindi = postprocess(raw_hindi, processed.entity_map)
            self._prev_hindi = hindi

            yield {"type": "transcript", "en": english, "hi": hindi, "stage": "final"}

            audio_bytes = await synthesize(hindi)
            if audio_bytes:
                yield {
                    "type": "audio",
                    "data": base64.b64encode(audio_bytes).decode(),
                }

        except Exception as exc:
            logger.exception("Pipeline error")
            yield {"type": "speech_rejected"}
            yield {"type": "error", "message": str(exc)}


def _audio_energy(samples: np.ndarray) -> tuple[float, float]:
    if samples.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    return rms, peak


def _is_low_energy_hallucination(text: str, rms: float) -> bool:
    if rms >= ASR_HALLUCINATION_RMS:
        return False
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    if normalized in ASR_HALLUCINATION_PHRASES:
        return True
    return bool(re.fullmatch(r"(thank you\s*)+", normalized))
