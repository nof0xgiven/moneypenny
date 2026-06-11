"""Sweep engine seeds for the offline regression scenario after the TTS
decoupling (briefing synthesized by BriefingSynth before engine construction).
For each seed: question -> inject_audio(briefing) -> 30s free-run; report
fact uptake ('31'), illusion break ('briefing'), and frames produced."""
import sys
from pathlib import Path

import sphn

sys.path.insert(0, str(Path(__file__).parent.parent))

from moneypenny.engine import VoiceEngine
from moneypenny.prompts import FRONT_OF_HOUSE
from moneypenny.tts import BriefingSynth

FRAME = 1920
QUESTION_WAV = Path(__file__).parent / "out" / "question.wav"
# The 2026-06-11 re-pin sweep ran these in two batches of five; 42424242 (the
# old pinned seed) appears twice to confirm the new trajectory is
# deterministic. Result: only 11 combined a normal (non-failsafe) gate open,
# the briefed "31" spoken (hedged), and no "briefing" said aloud; 42 had
# uptake but only via the 250-frame failsafe.
SEEDS = [42424242, 42424242, 7, 1234, 42, 2, 11, 123, 2024, 20260611]

briefing_pcm = BriefingSynth("am_michael").synthesize(
    "BRIEFING: WEATHER TODAY 31 CELSIUS CLEAR SKIES"
)
pcm, _ = sphn.read(str(QUESTION_WAV), sample_rate=24000)

for seed in SEEDS:
    eng = VoiceEngine(system_prompt=FRONT_OF_HOUSE, seed=seed)
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
    eng.inject_audio(briefing_pcm)
    drain(int(30 * 12.5))

    transcript = "".join(text_pieces)
    tl = transcript.lower()
    uptake = "31" in tl or "thirty-one" in tl or "thirty one" in tl
    illusion = "briefing" in tl
    print(f"\n=== seed={seed} uptake={uptake} illusion_break={illusion} "
          f"frames={len(out_frames)} gate_wait={eng.last_gate_wait_frames}")
    print(f"transcript: {transcript!r}", flush=True)
    del eng
