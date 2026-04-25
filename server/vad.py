"""
Voice Activity Detection using webrtcvad.

Accumulates raw 16kHz PCM Int16 frames from the WebSocket client.
Emits a complete phrase (as a bytes blob) whenever a long-enough
silence follows speech.
"""

import webrtcvad
import numpy as np
from collections import deque
from typing import Optional

from .config import (
    SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_SAMPLES,
    VAD_SILENCE_THRESHOLD_FRAMES,
    VAD_MIN_SPEECH_FRAMES,
)


class VADProcessor:
    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._speech_buf: list[bytes] = []
        self._silence_count = 0
        self._speech_count = 0
        self._in_speech = False
        # Keep a small pre-roll buffer so we don't clip the phrase start
        self._pre_roll: deque[bytes] = deque(maxlen=5)

    def feed(self, pcm_bytes: bytes) -> Optional[np.ndarray]:
        """
        Feed one chunk of raw PCM bytes (any size, will be split into 30ms frames).
        Returns a float32 numpy array (phrase audio at 16kHz) when a complete
        phrase is ready, otherwise None.
        """
        result: Optional[np.ndarray] = None

        # Split incoming bytes into 30ms frames
        frame_size = VAD_FRAME_SAMPLES * 2  # 2 bytes per int16 sample
        offset = 0
        while offset + frame_size <= len(pcm_bytes):
            frame = pcm_bytes[offset: offset + frame_size]
            offset += frame_size
            frame_result = self._process_frame(frame)
            if frame_result is not None:
                result = frame_result

        return result

    def _process_frame(self, frame: bytes) -> Optional[np.ndarray]:
        try:
            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_speech = False

        if is_speech:
            self._silence_count = 0
            self._speech_count += 1
            if not self._in_speech and self._speech_count >= VAD_MIN_SPEECH_FRAMES:
                self._in_speech = True
                # Prepend pre-roll so phrase start isn't clipped
                self._speech_buf = list(self._pre_roll)
            if self._in_speech:
                self._speech_buf.append(frame)
        else:
            self._pre_roll.append(frame)
            if self._in_speech:
                self._silence_count += 1
                self._speech_buf.append(frame)
                if self._silence_count >= VAD_SILENCE_THRESHOLD_FRAMES:
                    return self._flush()
            else:
                self._speech_count = 0

        return None

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
        return None

    def _reset(self) -> None:
        self._speech_buf = []
        self._silence_count = 0
        self._speech_count = 0
        self._in_speech = False
