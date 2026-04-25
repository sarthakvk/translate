"""
Voice Activity Detection using webrtcvad.

Accumulates raw 16kHz PCM Int16 frames from the WebSocket client.
Emits a complete phrase (as a bytes blob) whenever a long-enough
silence follows speech.
"""

import webrtcvad
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_SAMPLES,
    VAD_SILENCE_THRESHOLD_FRAMES,
    VAD_MIN_SPEECH_FRAMES,
    VAD_START_MIN_PEAK,
    VAD_START_MIN_RMS,
)


@dataclass
class VADEvent:
    speech_started: bool = False   # VAD just transitioned into in-speech this call
    phrase: Optional[np.ndarray] = field(default=None)  # completed phrase audio


class VADProcessor:
    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._speech_buf: list[bytes] = []
        self._candidate_buf: list[bytes] = []
        self._silence_count = 0
        self._speech_count = 0
        self._in_speech = False
        self._pending_bytes = b""
        # Keep a small pre-roll buffer so we don't clip the phrase start
        self._pre_roll: deque[bytes] = deque(maxlen=5)

    def feed(self, pcm_bytes: bytes) -> VADEvent:
        """
        Feed one chunk of raw PCM bytes (any size, will be split into 30ms frames).
        Returns a VADEvent with:
          - speech_started: True the first time VAD transitions into in-speech
          - phrase: float32 numpy array when a complete phrase is ready
        """
        speech_started = False
        phrase: Optional[np.ndarray] = None

        pcm_bytes = self._pending_bytes + pcm_bytes
        frame_size = VAD_FRAME_SAMPLES * 2  # 2 bytes per int16 sample
        offset = 0
        while offset + frame_size <= len(pcm_bytes):
            frame = pcm_bytes[offset: offset + frame_size]
            offset += frame_size
            started, result = self._process_frame(frame)
            if started:
                speech_started = True
            if result is not None:
                phrase = result

        self._pending_bytes = pcm_bytes[offset:]
        return VADEvent(speech_started=speech_started, phrase=phrase)

    def _process_frame(self, frame: bytes) -> tuple[bool, Optional[np.ndarray]]:
        """Returns (speech_started_this_frame, completed_phrase_or_None)."""
        try:
            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_speech = False

        speech_started = False

        if is_speech:
            self._silence_count = 0
            self._speech_count += 1
            if self._in_speech:
                self._speech_buf.append(frame)
            else:
                self._candidate_buf.append(frame)
                if self._speech_count >= VAD_MIN_SPEECH_FRAMES:
                    candidate = b"".join(self._candidate_buf)
                    rms, peak = _pcm_energy(candidate)
                    if rms < VAD_START_MIN_RMS and peak < VAD_START_MIN_PEAK:
                        self._speech_count = 0
                        self._candidate_buf = []
                        return speech_started, None

                    self._in_speech = True
                    speech_started = True
                    # Prepend ambient pre-roll plus the speech frames that confirmed
                    # the transition so the phrase start is not clipped.
                    self._speech_buf = list(self._pre_roll) + self._candidate_buf
                    self._candidate_buf = []
        else:
            self._pre_roll.append(frame)
            if self._in_speech:
                self._silence_count += 1
                self._speech_buf.append(frame)
                if self._silence_count >= VAD_SILENCE_THRESHOLD_FRAMES:
                    return speech_started, self._flush()
            else:
                self._speech_count = 0
                self._candidate_buf = []

        return speech_started, None

    def _flush(self) -> Optional[np.ndarray]:
        if not self._speech_buf:
            return None
        raw = b"".join(self._speech_buf)
        self._reset()
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

    def flush_pending(self) -> Optional[np.ndarray]:
        """Call when the client sends a stop signal to emit any buffered speech."""
        if self._in_speech and self._speech_buf:
            return self._flush()
        self._reset()
        self._pending_bytes = b""
        return None

    def _reset(self) -> None:
        self._speech_buf = []
        self._candidate_buf = []
        self._silence_count = 0
        self._speech_count = 0
        self._in_speech = False


def _pcm_energy(pcm_bytes: bytes) -> tuple[float, float]:
    if not pcm_bytes:
        return 0.0, 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    return rms, peak
