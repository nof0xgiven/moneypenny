"""Localize the worker-thread step penalty: time step() sub-components
(mimi encode / lm step / mimi decode) on the main thread vs a pool worker.

Usage: .venv/bin/python spikes/perf_step_breakdown.py --thread main|worker [--bits 8]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time

import numpy as np

from moneypenny.engine import FRAME, VoiceEngine
from moneypenny.prompts import FRONT_OF_HOUSE


def run(bits: int, steps: int, label: str) -> None:
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(FRAME) * 0.003).astype(np.float32)
    engine = VoiceEngine(system_prompt=FRONT_OF_HOUSE, quantize_bits=bits)
    for _ in range(10):
        engine.step(noise)

    enc, lm, dec = [], [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        tokens = engine._encode_pcm(noise)
        t1 = time.perf_counter()
        engine._gen.step(input_tokens=tokens)
        t2 = time.perf_counter()
        audio_tokens = engine._gen.last_audio_tokens()
        if audio_tokens is not None:
            decode = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            np.asarray(engine._mimi.decode_step(decode))
        t3 = time.perf_counter()
        enc.append((t1 - t0) * 1000)
        lm.append((t2 - t1) * 1000)
        dec.append((t3 - t2) * 1000)

    for name, samples in (("mimi_encode", enc), ("lm_step", lm), ("decode", dec)):
        s = sorted(samples)
        print(f"[part] {json.dumps({'label': f'{label} {name}', 'median_ms': round(statistics.median(s), 1), 'p90_ms': round(s[int(len(s) * 0.9) - 1], 1)})}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", choices=("main", "worker"), required=True)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    label = f"q{args.bits} {args.thread}"
    if args.thread == "main":
        run(args.bits, args.steps, label)
    else:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")
        pool.submit(run, args.bits, args.steps, label).result()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
