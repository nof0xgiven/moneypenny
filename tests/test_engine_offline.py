"""Offline engine test: WAV in -> briefing injected -> WAV + transcript out.
This is the engine's regression harness; it reuses the spike fixtures.

Stochasticity note: seeded generation with temp 0.7/0.8 - a sampler or weights
change can flip assertions. That is the point of a regression harness; if it
fails after an upstream change, listen to the WAV before blaming the test."""
import numpy as np
import pytest
import sphn

from moneypenny.engine import VoiceEngine
from moneypenny.prompts import FRONT_OF_HOUSE

FRAME = 1920


@pytest.mark.slow
def test_question_briefing_answer_cycle(tmp_path):
    eng = VoiceEngine(system_prompt=FRONT_OF_HOUSE, seed=42424242)
    pcm, _ = sphn.read("spikes/out/question.wav", sample_rate=24000)

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
    drain(int(20 * 12.5))  # gate waits for the hedge, ~5s briefing drains, then the answer

    transcript = "".join(text_pieces).lower()
    print(f"\ntranscript: {''.join(text_pieces)!r}")
    print(f"gate waited {eng.last_gate_wait_frames} frames before draining")
    # Fact uptake: the model must speak the briefed temperature
    assert "31" in transcript or "thirty-one" in transcript or "thirty one" in transcript
    # Illusion: never says the word "briefing"
    assert "briefing" not in transcript
    # Audio was produced
    assert len(out_frames) > 50
