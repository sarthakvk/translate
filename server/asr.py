"""
ASR via Groq Whisper API.

Accepts a float32 numpy array (16kHz mono) and returns the transcribed
English text. Audio is encoded to WAV in-memory — no temp files needed.
"""

import io
import wave
import asyncio
import numpy as np
from groq import AsyncGroq

from .config import GROQ_API_KEY, SAMPLE_RATE, WHISPER_MODEL, WHISPER_LANGUAGE

_client = AsyncGroq(api_key=GROQ_API_KEY)


def _to_wav_bytes(samples: np.ndarray) -> bytes:
    """Encode float32 PCM array to a WAV file in memory (int16)."""
    pcm_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


async def transcribe(samples: np.ndarray) -> str:
    """
    Transcribe audio samples to English text using Groq Whisper.

    Returns empty string if the audio is too short or silent.
    """
    duration_s = len(samples) / SAMPLE_RATE
    if duration_s < 0.3:
        return ""

    wav_bytes = _to_wav_bytes(samples)

    transcription = await _client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=("audio.wav", wav_bytes, "audio/wav"),
        language=WHISPER_LANGUAGE,
        response_format="text",
        # Prompt hint helps Whisper with proper nouns and formatting
        prompt="This is a spoken English sentence. Preserve proper nouns like Google Meet, Zoom, Slack, GitHub, Kubernetes, AWS.",
    )
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
