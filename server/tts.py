"""
Hindi Text-to-Speech using Microsoft edge-tts.

Synthesizes Hindi text into MP3 audio bytes asynchronously.
No API key required — edge-tts uses Microsoft's free TTS endpoint.
"""

import io
import edge_tts

from .config import TTS_VOICE


async def synthesize(text: str) -> bytes:
    """
    Synthesize Hindi text to MP3 bytes using edge-tts.
    Returns empty bytes if text is blank.
    """
    if not text.strip():
        return b""

    buf = io.BytesIO()
    communicate = edge_tts.Communicate(text, TTS_VOICE)

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])

    return buf.getvalue()
