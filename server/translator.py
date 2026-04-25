"""
English → Hindi translation using deep_translator (Google Translate backend).

No API key required for typical usage (free tier, unauthenticated).
Includes tone hinting and inter-phrase context continuity.
"""

import asyncio
import logging
from deep_translator import GoogleTranslator

from .preprocessor import ProcessedText


logger = logging.getLogger(__name__)
_translator = GoogleTranslator(source="en", target="hi")


async def translate(processed: ProcessedText, prev_hindi: str = "") -> str:
    """
    Translate pre-processed English text to Hindi.

    prev_hindi: the most recent Hindi output — used to maintain discourse continuity
    by briefly priming the translator (not exposed in output).
    """
    text = processed.cleaned
    if not text:
        return ""

    # Build a light context prefix that guides the translator without appearing in output.
    # Google Translate uses the full string as context, so we prepend prior output as a
    # leading clause and then strip it from the result.
    context_prefix = ""
    context_snippet = ""
    if prev_hindi:
        # Use the last ~60 chars of previous Hindi as soft context
        context_snippet = prev_hindi[-60:].strip()
        context_prefix = context_snippet + " | "

    input_text = context_prefix + text

    # Run blocking translator in a thread so we don't block the event loop
    loop = asyncio.get_running_loop()
    raw: str = await loop.run_in_executor(
        None, lambda: _translator.translate(input_text)
    )

    # Strip the context prefix from the result (it mirrors the Hindi prefix we injected)
    if context_prefix:
        sep = " | "
        if sep in raw:
            raw = raw[raw.index(sep) + len(sep):]
        else:
            logger.warning("Translator output did not preserve context separator")
            raw = _strip_context_prefix(raw, context_snippet)

    return raw.strip()


def _strip_context_prefix(raw: str, context_snippet: str) -> str:
    """Best-effort removal when Translate drops the explicit separator."""
    raw = raw.strip()
    context_snippet = context_snippet.strip()
    if not raw or not context_snippet:
        return raw

    if raw.startswith(context_snippet):
        return raw[len(context_snippet):]

    idx = raw.find(context_snippet)
    if 0 <= idx <= 20:
        return raw[idx + len(context_snippet):]

    max_prefix = min(len(raw), len(context_snippet))
    common = 0
    for common in range(max_prefix):
        if raw[common] != context_snippet[common]:
            break
    else:
        common = max_prefix

    if common >= min(20, len(context_snippet)):
        return raw[common:]

    return raw
