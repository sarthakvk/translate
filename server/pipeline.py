"""
Async orchestrator: VAD phrase → ASR → preprocess → translate → TTS.

Each WebSocket connection owns one Pipeline instance. The pipeline exposes:
  - feed(pcm_bytes) → async generator yielding WS messages
  - stop()          → flushes any buffered speech and yields final messages

Audio is queued on the client side — new phrases never cancel in-flight TTS.

WS message shapes (dicts, serialised to JSON by main.py):
  {"type": "transcript", "en": str, "stage": "final"}
  {"type": "transcript", "en": str, "hi": str, "stage": "final"}
  {"type": "audio",      "data": <base64 MP3 str>}
  {"type": "error",      "message": str}
"""

import base64
import logging
from typing import AsyncGenerator

import numpy as np

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

    async def feed(self, pcm_bytes: bytes) -> AsyncGenerator[dict, None]:
        event = self._vad.feed(pcm_bytes)
        if event.speech_started:
            yield {"type": "speech_start"}
        if event.phrase is not None:
            # Resume paused audio immediately — before the slow ASR/translate/TTS pipeline
            yield {"type": "speech_end"}
            async for msg in self._run(event.phrase):
                yield msg

    async def stop(self) -> AsyncGenerator[dict, None]:
        phrase = self._vad.flush_pending()
        if phrase is not None:
            async for msg in self._run(phrase):
                yield msg

    async def _run(self, samples: np.ndarray) -> AsyncGenerator[dict, None]:
        try:
            english = await transcribe(samples)
            if not english:
                return

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
            yield {"type": "error", "message": str(exc)}
