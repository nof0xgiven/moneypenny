"""The decisive experiment behind the import-affinity fix (decision 0002, KL6).

Two configurations of the SAME work — build VoiceEngine on a 1-worker pool and
time 100 step(noise) calls:

  --imports main    mlx stack imported on the MAIN thread first (the app's old
                    shape). Measured: ~183-188ms median step with the original
                    synchronous mimi decode; ~143ms with the now-committed
                    pipelined decode.
  --imports worker  mlx stack imported ON the engine worker (the app's new
                    shape, app._load_engine). Measured: ~104ms median step
                    synchronous-decode; ~60ms with the pipelined decode.

Profiling (macOS `sample`) showed the slow case burning the difference in
malloc/free churn inside mlx eval. Thread QoS (pthread_set_qos_class_self_np
before/after MLX init), MLX stream ids (worker owning Stream(gpu, 0) was still
slow when imports were on main), pure-MLX micro-benchmarks (no penalty on any
thread), and run_in_executor overhead (~3ms) were each ruled out by direct
measurement on this machine (M3 Ultra, 96GB, macOS 25.5, mlx 0.31.2).

Usage: .venv/bin/python spikes/perf_import_affinity.py --imports main|worker
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time

import numpy as np

FRAME = 1920


def build_and_time(steps: int, label: str) -> None:
    # Imports here so the CALLING thread is the first mlx importer.
    from moneypenny.engine import VoiceEngine
    from moneypenny.prompts import FRONT_OF_HOUSE

    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(FRAME) * 0.003).astype(np.float32)
    engine = VoiceEngine(system_prompt=FRONT_OF_HOUSE, quantize_bits=8)
    for _ in range(10):
        engine.step(noise)
    samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        engine.step(noise)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    print(f"[steps] {json.dumps({'label': label, 'n': steps, 'median_ms': round(statistics.median(samples), 1), 'p90_ms': round(samples[int(steps * 0.9) - 1], 1)})}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imports", choices=("main", "worker"), required=True)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    if args.imports == "main":
        import moneypenny.engine  # noqa: F401  (first mlx import: main thread)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")
    pool.submit(build_and_time, args.steps, f"imports-on-{args.imports}").result()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
