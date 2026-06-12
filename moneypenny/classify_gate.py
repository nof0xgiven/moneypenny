"""Pre-router transcript gate: is this ASR transcript worth classifying?

Pure stdlib (no mlx, no numpy): moneypenny.app imports this at module level,
so it must stay clear of the MLX import-affinity rule
(tests/test_app_imports.py).

Why: the frame loop classifies the ASR partial on every VAD maybe_end (a
160ms pause) and again on utterance_end when the transcript grew. Live
sessions showed three failure modes:
  - backchannel spam: "Yeah." / "Oh." / "Uh uh." each burned a ~200ms route
    worker classification (~11 router calls in 7s during one exchange);
  - duplicates: the same partial classified twice back-to-back (two
    maybe_ends with no transcript change);
  - self-echo phantoms (the worst): on speakers with no echo cancellation,
    the model's own speech leaks into the mic and comes back as phantom
    "user" utterances that escalate tier 2 and can trigger tools. We KNOW
    what the model is saying (its text streams from engine.step), so a
    transcript-similarity filter drops these before routing.

Four checks, cheapest first; the first hit wins (should_classify reasons):
  empty       no alphanumeric tokens at all
  backchannel EVERY token is in BACKCHANNEL_TOKENS — a single real content
              word ("lights off", "stop the timer") defeats the filter,
              which is the mechanism: pure filler blocks, anything
              resembling a command passes
  duplicate   same normalized token sequence as the last ALLOWED transcript
              of the current utterance (cleared by reset_utterance on
              speech_start)
  self_echo   >= echo_overlap_threshold of the transcript's content tokens
              (stopwords = the backchannel set) appear in what the model
              spoke during the last echo_window_s seconds, and there are at
              least 2 content tokens to judge by

Self-echo matching detail: the model's text stream mangles words ("I don
havet that in front of me") and ASR mangles them differently again ("I
don't have that channel"), and engine.step pieces can split a word across
calls ("chan" + "nel"). So a transcript token also matches a window word
when one is a prefix of the other and both halves of the comparison are at
least _MIN_PREFIX_MATCH chars; each window word is consumable once per
evaluation (multiset, so a repeated transcript word needs repeated model
evidence).

Known limitation (accepted): a user genuinely repeating the model's words
within the window gets dropped, and a phantom mangled beyond the prefix
rule (or sharing too few tokens, like the log's "I can't check that thing
right now.") still gets through. Echo cancellation is the real fix; this is
the cheap second layer that keeps the obvious phantoms away from the router
and the tool host.
"""
from __future__ import annotations

from collections import Counter, deque

# Tokens that count as pure conversational filler when they make up the
# WHOLE transcript; doubles as the stopword set for self-echo content
# selection. Tuned against the live log: "Yeah.", "Oh my god.", "Uh uh.",
# "Yeah, that's right.", "Yeah, I know." block, while "lights off",
# "what's the weather", "stop the timer" pass on their content words.
BACKCHANNEL_TOKENS = frozenset({
    "yeah", "yea", "yep", "yes", "no", "nope", "ok", "okay", "oh", "ah",
    "uh", "um", "huh", "hm", "hmm", "mm", "mhm", "right", "sure", "cool",
    "fine", "thanks", "thank", "you", "alright", "wow", "god", "my",
    "that's", "thats", "like", "so", "well", "i", "know", "see", "good",
    "great", "totally", "exactly",
})

# "dont"~"don", "havet"~"have", "channel"~"chan"; 1-2 char tokens must match
# exactly (prefix matching that short would make "i"/"it"/"in" interchangeable).
_MIN_PREFIX_MATCH = 3


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Punctuation — apostrophes included —
    is stripped, so "That's" -> "thats" and "don't" -> "dont"."""
    out = []
    for raw in text.lower().split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if tok:
            out.append(tok)
    return out


# Membership tests run on normalized tokens, so normalize the allowlist once:
# both spellings of a contraction ("that's"/"thats") collapse to one entry.
_STOPWORDS = frozenset(tok for entry in BACKCHANNEL_TOKENS for tok in _tokens(entry))


def _prefix_match(a: str, b: str) -> bool:
    if min(len(a), len(b)) < _MIN_PREFIX_MATCH:
        return False
    return a.startswith(b) or b.startswith(a)


class ClassifyGate:
    """Decides whether an ASR transcript is worth routing (module docstring
    has the full filter semantics). Single-threaded by design: lives on the
    frame-loop thread, which is the only caller of every method."""

    def __init__(self, echo_window_s: float = 10.0,
                 echo_overlap_threshold: float = 0.7) -> None:
        self._window_s = echo_window_s
        self._threshold = echo_overlap_threshold
        self._model_words: deque[tuple[str, float]] = deque()  # (word, noted-at)
        self._last_allowed: str | None = None  # normalized; per-utterance

    def note_model_text(self, piece: str, now: float) -> None:
        """Feed every text piece the model speaks (engine.step output)."""
        for word in _tokens(piece):
            self._model_words.append((word, now))
        self._expire(now)

    def reset_utterance(self) -> None:
        """New utterance began (speech_start): clear the dedupe memory ONLY.
        The echo window must survive utterance boundaries — the phantom
        arrives as its own 'utterance' after the model spoke."""
        self._last_allowed = None

    def should_classify(self, transcript: str, now: float) -> tuple[bool, str]:
        """(allow, reason); reason in {"ok", "empty", "backchannel",
        "duplicate", "self_echo"}. Allowed transcripts become the dedupe
        memory; blocked ones leave it untouched."""
        toks = _tokens(transcript)
        if not toks:
            return False, "empty"
        if all(t in _STOPWORDS for t in toks):
            return False, "backchannel"
        normalized = " ".join(toks)
        if normalized == self._last_allowed:
            return False, "duplicate"
        if self._is_echo(toks, now):
            return False, "self_echo"
        self._last_allowed = normalized
        return True, "ok"

    def _expire(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._model_words and self._model_words[0][1] < cutoff:
            self._model_words.popleft()

    def _is_echo(self, toks: list[str], now: float) -> bool:
        content = [t for t in toks if t not in _STOPWORDS]
        if len(content) < 2:
            return False  # too little signal: a lone "Weather." is not echo evidence
        self._expire(now)
        if not self._model_words:
            return False
        window = Counter(word for word, _ in self._model_words)
        matched = 0
        for tok in content:
            if window[tok] > 0:
                window[tok] -= 1
                matched += 1
                continue
            for word, count in window.items():
                if count > 0 and _prefix_match(tok, word):
                    window[word] -= 1
                    matched += 1
                    break
        return matched / len(content) >= self._threshold
