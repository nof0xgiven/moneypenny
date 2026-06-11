"""Engine thread affinity: MLX streams are thread-bound (per-thread stream
registries since mlx 0.31), so a VoiceEngine constructed on one thread cannot
be stepped on another — evaluation dies with
"RuntimeError: There is no Stream(gpu, N) in current thread." followed by a
bus error. That is exactly what the first live run hit: app.py built the
engine on the event-loop thread and ran step() on the engine worker.

This test replicates the app's threading pattern WITHOUT audio devices and
asserts the FIXED contract: construct the engine ON the single engine worker
and run every step()/inject() on that same worker. (The broken
construct-on-main pattern was reproduced once during diagnosis and is not
kept as a test — it crashes the process with SIGBUS after the RuntimeError.)
"""
import concurrent.futures

import pytest

from moneypenny.engine import VoiceEngine
from moneypenny.prompts import FRONT_OF_HOUSE


@pytest.mark.slow
def test_engine_constructed_and_stepped_on_same_worker_thread():
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="engine"
    ) as pool:
        engine = pool.submit(
            lambda: VoiceEngine(system_prompt=FRONT_OF_HOUSE, seed=42424242)
        ).result()

        out_frames = []

        def step_on_worker(n: int) -> None:
            for _ in range(n):
                audio_out, _ = pool.submit(engine.step, None).result()
                if audio_out is not None:
                    out_frames.append(audio_out)

        # ~2s of frame loop, then a briefing injection (TTS on the worker,
        # like app.py), then a few more frames — the live crash fired on the
        # very first step, so surviving all of this is the regression bar.
        step_on_worker(25)
        pool.submit(engine.inject, "BRIEFING: TEST").result()
        step_on_worker(5)

        assert out_frames, "engine produced no audio frames across 30 steps"
