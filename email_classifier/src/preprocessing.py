"""Text preprocessing utilities for email classification."""
from __future__ import annotations

import re
from typing import Iterable, Optional

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Normalize raw email text into a compact lowercase representation."""
    if text is None:
        return ""

    normalized = str(text)
    if not normalized or normalized.lower() in {"nan", "none", "null"}:
        return ""

    normalized = _HTML_TAG_RE.sub(" ", normalized)
    normalized = _URL_RE.sub(" ", normalized)
    normalized = normalized.lower()
    normalized = _NON_WORD_RE.sub(" ", normalized)
    normalized = _MULTI_SPACE_RE.sub(" ", normalized).strip()
    return normalized


def clean_texts(texts: Iterable[object]) -> list[str]:
    """Vector-friendly helper for cleaning an iterable of text values."""
    return [clean_text(text) for text in texts]


def safe_preview(text: Optional[object], max_length: int = 120) -> str:
    """Short preview for UI display."""
    cleaned = clean_text(text)
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[:max_length].rstrip()}..."
