"""H3 + context-growth probe for the live-vs-offline step-time gap.

H3: time engine.step(noise) called directly on the worker thread vs awaited
through loop.run_in_executor on the same 1-worker pool (engine constructed on
that worker, matching the app's thread-affinity rule).

H4 (context growth): the live app has stepped thousands of frames; attention
cost grows with sequence length. Step 4000 extra frames and re-time at
intervals.

Usage: .venv/bin/python spikes/perf_executor_and_context.py [--bits 8]
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import statistics
import time

import numpy as np

from moneypenny.engine import FRAME, VoiceEngine
from moneypenny.prompts import FRONT_OF_HOUSE


def summarize(samples: list[float], label: str) -> dict:
    samples = sorted(samples)
    stats = {
        "label": label,
        "n": len(samples),
        "median_ms": round(statistics.median(samples), 1),
        "p90_ms": round(samples[int(len(samples) * 0.9) - 1], 1),
        "min_ms": round(samples[0], 1),
        "max_ms": round(samples[-1], 1),
    }
    print(f"[steps] {json.dumps(stats)}", flush=True)
    return stats


def time_direct(engine: VoiceEngine, noise: np.ndarray, n: int, label: str) -> None:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        engine.step(noise)
        samples.append((time.perf_counter() - t0) * 1000)
    summarize(samples, label)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(FRAME) * 0.003).astype(np.float32)

    loop = asyncio.get_running_loop()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")
    engine = await loop.run_in_executor(
        pool, lambda: VoiceEngine(system_prompt=FRONT_OF_HOUSE, quantize_bits=args.bits)
    )
    print(f"[load] engine q{args.bits} ready", flush=True)

    # warmup
    for _ in range(10):
        await loop.run_in_executor(pool, engine.step, noise)

    # H3a: direct tight loop ON the worker thread
    await loop.run_in_executor(
        pool, time_direct, engine, noise, args.steps, f"H3a q{args.bits} direct-on-worker"
    )

    # H3b: per-step await through run_in_executor (the app's pattern)
    samples = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        await loop.run_in_executor(pool, engine.step, noise)
        samples.append((time.perf_counter() - t0) * 1000)
    summarize(samples, f"H3b q{args.bits} via-run_in_executor")

    # H4: context growth - free-run and re-time at increasing sequence position
    pos = args.steps * 2 + 20
    for chunk in (500, 1000, 1500, 2000):
        def advance(n: int = chunk) -> None:
            for _ in range(n):
                engine.step(noise)
        t0 = time.perf_counter()
        await loop.run_in_executor(pool, advance)
        pos += chunk
        print(f"[advance] +{chunk} frames in {time.perf_counter() - t0:.0f}s", flush=True)
        await loop.run_in_executor(
            pool, time_direct, engine, noise, 50, f"H4 q{args.bits} at ~{pos} frames (~{pos / 12.5:.0f}s)"
        )
        pos += 50
    print("[done]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
