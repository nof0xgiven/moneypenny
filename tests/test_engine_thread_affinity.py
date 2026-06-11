"""Engine thread affinity: MLX streams are thread-bound (per-thread stream
registries since mlx 0.31), so a VoiceEngine must be constructed and used on
the SAME thread. This test pins the supported pattern — construct + step +
inject all on one engine worker, mirroring app.py — without audio devices.

No negative test exists on purpose: constructing on one thread and stepping
on another raises RuntimeError ("There is no Stream(gpu, N) in current
thread.") and can SIGBUS the whole process.
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
        # like app.py), then a few more frames: covers every engine entry
        # point the app exercises across a thread boundary.
        step_on_worker(25)
        pool.submit(engine.inject, "BRIEFING: TEST").result()
        step_on_worker(5)

        assert out_frames, "engine produced no audio frames across 30 steps"
