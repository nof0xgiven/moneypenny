"""Offline engine test: WAV in -> briefing injected -> WAV + transcript out.
This is the engine's regression harness; it reuses the spike fixtures.

Stochasticity note: seeded generation with temp 0.7/0.8 - a sampler or weights
change can flip assertions. That is the point of a regression harness; if it
fails after an upstream change, listen to the WAV before blaming the test."""
from pathlib import Path

import numpy as np
import pytest
import sphn

from moneypenny.engine import INJECT_AFTER_QUIET_FRAMES, VoiceEngine
from moneypenny.prompts import FRONT_OF_HOUSE

FRAME = 1920
QUESTION_WAV = Path(__file__).parent.parent / "spikes" / "out" / "question.wav"


def _gate_shell() -> VoiceEngine:
    """Engine with only the injection-gate state initialized: the gate is a
    pure state machine that needs no model, so it can be tested fast."""
    eng = VoiceEngine.__new__(VoiceEngine)
    eng._pending_audio = None
    eng._draining = False
    eng._quiet_frames = 0
    eng._inject_waited = 0
    eng.last_gate_wait_frames = None
    return eng


def test_gate_resets_between_briefings():
    eng = _gate_shell()
    eng.inject_audio(np.ones(FRAME * 2, dtype=np.float32))

    # Gate closed: nothing drains, the failsafe counter ticks.
    assert eng._gate_and_drain() is None
    assert eng._inject_waited == 1

    # Output quiet long enough: drain starts and runs to completion,
    # and a briefing injected MID-drain appends without re-gating.
    eng._quiet_frames = INJECT_AFTER_QUIET_FRAMES
    assert eng._gate_and_drain() is not None
    eng.inject_audio(np.ones(FRAME, dtype=np.float32))
    assert eng._gate_and_drain() is not None
    assert eng._gate_and_drain() is not None
    assert eng._pending_audio is None

    # Completion must reset the gate so the NEXT briefing re-gates instead
    # of firing instantly off the stale quiet streak.
    assert eng._quiet_frames == 0
    assert eng._inject_waited == 0
    eng.inject_audio(np.ones(FRAME, dtype=np.float32))
    assert eng._gate_and_drain() is None


@pytest.mark.slow
def test_question_briefing_answer_cycle(tmp_path):
    eng = VoiceEngine(system_prompt=FRONT_OF_HOUSE, seed=42424242)
    pcm, _ = sphn.read(str(QUESTION_WAV), sample_rate=24000)

    out_frames = []
    text_pieces = []

    def drain(n_frames, mic=None):
        for i in range(n_frames):
            frame = (
                mic[0, i * FRAME:(i + 1) * FRAME]
                if mic is not None and (i + 1) * FRAME <= mic.shape[-1]
                else None
            )
            audio_out, text = eng.step(frame)
            if audio_out is not None:
                out_frames.append(audio_out)
            if text:
                text_pieces.append(text)

    drain(pcm.shape[-1] // FRAME + 1, mic=pcm)
    eng.inject("BRIEFING: WEATHER TODAY 31 CELSIUS CLEAR SKIES")
    # The model's uninterrupted hedge runs ~16s before the 2s-quiet gate opens
    # (~18s), the ~4s briefing drains, then the answer lands ~23-26s: 30s of
    # free-run covers the full cycle with headroom.
    drain(int(30 * 12.5))

    transcript = "".join(text_pieces).lower()
    print(f"\ntranscript: {''.join(text_pieces)!r}")
    print(f"gate waited {eng.last_gate_wait_frames} frames before draining")
    # Fact uptake: the model must speak the briefed temperature
    assert "31" in transcript or "thirty-one" in transcript or "thirty one" in transcript
    # Illusion: never says the word "briefing"
    assert "briefing" not in transcript
    # Audio was produced
    assert len(out_frames) > 50
