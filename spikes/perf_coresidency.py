"""H1/H2 perf probe: does co-residency (router + ASR) degrade engine.step?

One process. Stage A: VoiceEngine only (Kokoro is inside it), time 100
step(None) and 100 step(noise). Stage B: load Router + StreamingTranscriber
into the same process, re-time the same steps. Memory recorded at each stage.

Usage: .venv/bin/python spikes/perf_coresidency.py [--bits 8] [--steps 100]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx
import numpy as np

from moneypenny.asr import StreamingTranscriber
from moneypenny.engine import FRAME, VoiceEngine
from moneypenny.prompts import FRONT_OF_HOUSE
from moneypenny.router import Router


def mem_snapshot(label: str) -> dict:
    snap = {
        "label": label,
        "active_gb": round(mx.get_active_memory() / 2**30, 2),
        "peak_gb": round(mx.get_peak_memory() / 2**30, 2),
    }
    print(f"[mem] {json.dumps(snap)}", flush=True)
    return snap


def time_steps(engine: VoiceEngine, n: int, mic: np.ndarray | None, label: str) -> dict:
    for _ in range(10):  # warmup
        engine.step(mic)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        engine.step(mic)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    stats = {
        "label": label,
        "n": n,
        "median_ms": round(statistics.median(samples), 1),
        "p90_ms": round(samples[int(n * 0.9) - 1], 1),
        "mean_ms": round(statistics.fmean(samples), 1),
        "min_ms": round(samples[0], 1),
        "max_ms": round(samples[-1], 1),
    }
    print(f"[steps] {json.dumps(stats)}", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(FRAME) * 0.003).astype(np.float32)  # room-tone level

    info = mx.device_info()
    print(f"[device] {json.dumps({k: str(v) for k, v in info.items()})}", flush=True)

    t0 = time.perf_counter()
    engine = VoiceEngine(system_prompt=FRONT_OF_HOUSE, quantize_bits=args.bits)
    print(f"[load] engine q{args.bits} loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    mem_snapshot("after engine load")

    time_steps(engine, args.steps, None, f"A q{args.bits} engine-only step(None)")
    time_steps(engine, args.steps, noise, f"A q{args.bits} engine-only step(noise)")
    mem_snapshot("after stage A")

    t0 = time.perf_counter()
    router = Router()
    asr = StreamingTranscriber()
    print(f"[load] router+asr loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    # Exercise both once so lazy weights/buffers materialize as in a live session.
    router.classify("what's the weather today")
    asr.add_frame(noise)
    mem_snapshot("after router+asr load")

    time_steps(engine, args.steps, None, f"B q{args.bits} co-resident step(None)")
    time_steps(engine, args.steps, noise, f"B q{args.bits} co-resident step(noise)")
    mem_snapshot("after stage B")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
