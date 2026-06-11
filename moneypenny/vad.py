"""Energy-based utterance boundary detection on 80ms frames.

Deliberately dumb: RMS threshold + trailing-silence counts. Parakeet partials
provide the words; this only provides boundaries:
  maybe_end     - soft boundary (default 160ms silence): start classifying (P0.1)
  utterance_end - hard boundary (default 640ms): utterance confirmed over
Replace with a model VAD only if real-mic testing shows it failing."""
from __future__ import annotations

import numpy as np


class EnergyVAD:
    def __init__(
        self,
        rms_threshold: float = 0.01,
        soft_silence_frames: int = 2,
        silence_frames: int = 8,
    ) -> None:
        assert soft_silence_frames < silence_frames
        self._threshold = rms_threshold
        self._soft = soft_silence_frames
        self._hard = silence_frames
        self._in_speech = False
        self._silent_run = 0

    def feed(self, frame: np.ndarray) -> str | None:
        """Returns 'speech_start', 'maybe_end', 'utterance_end', or None."""
        is_loud = float(np.sqrt(np.mean(frame**2))) >= self._threshold
        if not self._in_speech:
            if is_loud:
                self._in_speech = True
                self._silent_run = 0
                return "speech_start"
            return None
        if is_loud:
            self._silent_run = 0
            return None
        self._silent_run += 1
        if self._silent_run == self._soft:
            return "maybe_end"
        if self._silent_run >= self._hard:
            self._in_speech = False
            self._silent_run = 0
            return "utterance_end"
        return None
