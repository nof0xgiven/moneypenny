"""VAD-gated ASR feeding policy: decides, per 80ms frame, what (if anything)
goes to the streaming transcriber.

Ambient silence is most of a session; running parakeet on every frame starves
the frame loop (GPU contention with engine.step) and lets the mic queue drift
unboundedly behind real time. The gate makes silent frames cost ZERO ASR work:

- Gated off: frames are NOT transcribed; the last `preroll_frames` raw frames
  are kept in a ring buffer so the next utterance's word onset isn't clipped.
- speech_start: the pre-roll (up to 320ms) plus the current frame are flushed
  to the transcriber in one call.
- While speaking: frames are batched in PAIRS (one buffered, then one call
  with 3840 samples) to halve parakeet encoder passes. The partial transcript
  therefore lags at most one frame (160ms total) - within P0.1's <200ms.
- Hangover: feeding continues for the VAD's hard boundary worth of silence
  (8 frames / 640ms with EnergyVAD defaults) so trailing words land before
  `utterance_end` triggers finish(). The gate has no hangover counter of its
  own: `in_speech` stays True through exactly those 8 silent frames and
  `utterance_end` arrives on the 8th, where any buffered frame is force-
  flushed together with the final one.

The gate is pure frame bookkeeping (no model access): the caller feeds it the
raw frame plus the EnergyVAD event/in_speech for that same frame, and sends
any returned buffer to StreamingTranscriber.add_frame (resample_poly and
parakeet's add_audio both accept arbitrary lengths).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class AsrGate:
    def __init__(self, preroll_frames: int = 4) -> None:
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self._pending: np.ndarray | None = None
        self._active = False

    @property
    def active(self) -> bool:
        """True from speech_start through the end of the hangover (the
        utterance_end frame). Drives the status line and the catch-up guard:
        frames must never be dropped while this is True."""
        return self._active

    def feed(self, frame: np.ndarray, event: str | None, in_speech: bool) -> np.ndarray | None:
        """Returns samples to transcribe now, or None (no ASR call this frame)."""
        if event == "speech_start":
            self._active = True
            buf = np.concatenate([*self._preroll, frame])
            self._preroll.clear()
            self._pending = None
            return buf
        if event == "utterance_end":
            # 8th hangover frame: force-flush so finish() sees the full tail
            self._active = False
            parts = [self._pending, frame] if self._pending is not None else [frame]
            self._pending = None
            return np.concatenate(parts)
        if in_speech:  # includes hangover frames 1..7 (in_speech stays True)
            if self._pending is None:
                self._pending = frame
                return None
            buf = np.concatenate([self._pending, frame])
            self._pending = None
            return buf
        self._preroll.append(frame)  # silence: keep pre-roll fresh for next utterance
        return None
