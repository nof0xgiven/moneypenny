"""Briefing TTS (Kokoro), run OFF the engine worker.

Phase 1 ran Kokoro synthesis inside VoiceEngine.inject() on the engine worker:
each briefing stalled the frame loop ~0.4-3s (no PCM produced -> speaker queue
drained -> audible underruns) and permanently ratcheted speaker latency via
the post-stall catch-up burst. BriefingSynth owns Kokoro instead; the app
constructs and calls it on a dedicated 1-worker pool so synthesis overlaps
engine.step, and only the finished PCM crosses to the engine worker via
inject_audio().

Thread affinity: MLX components must be constructed on the thread that runs
them (same rule as engine/router/asr - see app.py module docstring), so the
app constructs BriefingSynth on the tts pool worker. This module is imported
inside the app's loader function, never at app module level (import-affinity
rule: the engine worker must win the first mlx import).

RNG: synthesize() seeds TTS_SEED so briefing audio stays byte-identical for a
given text (uptake variance across renderings is open risk 1 of decision
0001). It does NOT snapshot/restore the surrounding RNG state any more: the
engine's seeded-trajectory concern only exists in offline tests, and those
synthesize briefings BEFORE constructing/seeding the engine (restoring global
RNG state from this thread while the engine samples concurrently would be a
race, not protection).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import sphn
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model as load_tts_model

from personaplex_mlx.persona_utils import seed_all

SAMPLE_RATE = 24000
KOKORO_MODEL = "prince-canuma/Kokoro-82M"

# Fixed RNG seed for briefing synthesis: makes briefing audio byte-identical
# for a given text, independent of session state at synthesis time. The value
# was selected by sweeping seeds against the offline regression scenario (the
# briefing waveform is the only free variable in an otherwise deterministic
# trajectory; seeds 0-1 yielded smalltalk deflection, 2 yielded clean fact
# uptake). Uptake variance across briefing renderings is open risk 1 of
# docs/decisions/0001 and is tracked into the live tests.
TTS_SEED = 2


class BriefingSynth:
    """Kokoro wrapper: text -> 24kHz mono float32 PCM, one voice."""

    def __init__(self, voice: str = "am_michael") -> None:
        self._voice = voice
        self._kokoro = load_tts_model(model_path=KOKORO_MODEL)
        # Fail fast on a typo'd voice: generate_audio swallows a bad-voice
        # load error (prints it and writes no wav), so without this probe a
        # typo only surfaces as a runtime synthesis failure - briefings
        # silently never play.
        try:
            self.synthesize("CHECK")
        except Exception as exc:
            raise RuntimeError(
                f"briefing voice {voice!r} failed the startup synthesis "
                "probe; check BRIEFING_VOICE against the Kokoro voice list"
            ) from exc

    def synthesize(self, text: str) -> np.ndarray:
        """TTS briefing text to 24kHz mono float32 PCM."""
        seed_all(TTS_SEED)
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix = Path(tmp_dir) / "briefing"
            generate_audio(
                text=text,
                model=self._kokoro,
                voice=self._voice,
                file_prefix=str(prefix),
                audio_format="wav",
                join_audio=True,
                play=False,
                verbose=False,
            )
            # Normalize to 24kHz mono float32 (Kokoro emits 24kHz, but sphn
            # resampling makes that an invariant rather than an assumption).
            pcm, _ = sphn.read(str(prefix.with_suffix(".wav")), sample_rate=SAMPLE_RATE)
        return pcm[0].astype(np.float32)
