"""
Pre- and post-processing for the translation pipeline.

Pre-processing (English text before translation):
  1. Strip filler words ("uh", "you know", etc.)
  2. Remove word stutters ("the the meeting" → "the meeting")
  3. Tag named entities so they survive translation verbatim

Post-processing (Hindi text after translation):
  1. Restore any named entities the translator may have mangled
  2. Clean up double spaces / punctuation artefacts
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .config import FILLERS, NAMED_ENTITIES

# Placeholder format used to protect entities during translation
_PLACEHOLDER_TEMPLATE = "XENTX{idx}X"
_PLACEHOLDER_RE = re.compile(r"XENTX(\d+)X")

# Pre-sorted and pre-compiled at import time — never re-sorted or re-compiled per phrase
_FILLER_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)\b" + re.escape(f) + r"\b[,]?\s*")
    for f in sorted(FILLERS, key=len, reverse=True)
]
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    (entity, re.compile(re.escape(entity), re.IGNORECASE))
    for entity in sorted(NAMED_ENTITIES, key=len, reverse=True)
]


@dataclass
class ProcessedText:
    cleaned: str                        # text to send to translator
    entity_map: dict[int, str] = field(default_factory=dict)  # idx → original entity
    tone: str = "neutral"               # "formal" | "casual" | "neutral"


def preprocess(text: str) -> ProcessedText:
    text = _strip_fillers(text)
    text = _collapse_stutters(text)
    text, entity_map = _tag_entities(text)
    tone = _detect_tone(text)
    return ProcessedText(cleaned=text.strip(), entity_map=entity_map, tone=tone)


def postprocess(hindi: str, entity_map: dict[int, str]) -> str:
    def restore(m: re.Match) -> str:
        idx = int(m.group(1))
        return entity_map.get(idx, m.group(0))

    hindi = _PLACEHOLDER_RE.sub(restore, hindi)
    hindi = re.sub(r" {2,}", " ", hindi).strip()
    return hindi


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _strip_fillers(text: str) -> str:
    for pattern in _FILLER_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _collapse_stutters(text: str) -> str:
    """Remove immediate word repetitions: 'the the meeting' → 'the meeting'."""
    return re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)


def _tag_entities(text: str) -> tuple[str, dict[int, str]]:
    """
    Replace named entities with placeholders so Google Translate won't mangle them.
    Patterns are pre-sorted longest-first to avoid partial matches (e.g. "Google"
    matching inside "Google Meet").
    """
    entity_map: dict[int, str] = {}
    idx = 0
    for entity, pattern in _ENTITY_PATTERNS:
        match = pattern.search(text)
        if match:
            placeholder = _PLACEHOLDER_TEMPLATE.format(idx=idx)
            entity_map[idx] = match.group(0)
            text = pattern.sub(placeholder, text)
            idx += 1
    return text, entity_map


def _detect_tone(text: str) -> str:
    from .config import FORMAL_MARKERS, CASUAL_MARKERS
    words = set(re.findall(r"\b\w+\b", text.lower()))
    formal_score = len(words & FORMAL_MARKERS)
    casual_score = len(words & CASUAL_MARKERS)
    if formal_score > casual_score:
        return "formal"
    if casual_score > formal_score:
        return "casual"
    return "neutral"
