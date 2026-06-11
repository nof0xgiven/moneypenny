"""Terse earpiece briefings. Pure functions, no I/O.

Format contract (spec sect. 8): prefix-tagged, uppercase, <= 40 content words.
Uppercase + telegraphic style optimizes TTS speed and model comprehension; the
voice model paraphrases, so style here never reaches the listener.
"""
from __future__ import annotations

MAX_WORDS = 40

_PREFIXES = {
    "briefing": "BRIEFING",
    "correction": "CORRECTION",
    "offline": "OFFLINE",
    "deferred": "DEFERRED",
}


def compose(kind: str, text: str) -> str:
    try:
        prefix = _PREFIXES[kind]
    except KeyError:
        raise ValueError(f"unknown briefing kind: {kind!r}") from None
    words = text.upper().split()
    return f"{prefix}: " + " ".join(words[:MAX_WORDS])
