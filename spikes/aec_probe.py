"""AEC library evidence harness: convergence, double-talk, and per-frame cost.

Synthesizes a known echo path (bulk delay + early reflections + speaker HF
rolloff), plays a real TTS fixture through it as the far/model signal, mixes
in a second fixture as near/user speech (double-talk), and runs a candidate
echo canceller frame-by-frame at the app's framing (1920 samples @ 24kHz,
chunked to the canceller's native frame internally). Reports:

  - ERLE (echo return loss enhancement, dB) per 0.5s window in echo-only
    regions: 10*log10(mic_power/out_power). Higher = more echo removed.
  - time-to-10dB: first window where ERLE crosses 10dB (convergence speed).
  - converged ERLE: mean windowed ERLE over the last echo-only stretch
    before near speech starts.
  - double-talk near preservation: peak normalized correlation of canceller
    output vs the clean near signal over the near-speech span, searched over
    +-20ms lag (the preprocessor's FFT block delay is not damage). ~1.0 =
    user speech survives.
  - per-1920-frame process() wall time (mean/p99/max ms) including the
    float32<->int16 conversion the app pays.

Implementations probed (whichever import):
  - app:        moneypenny.aec.EchoCanceller, the shipped one (ctypes ->
                libspeexdsp, MDF filter + preprocessor residual suppression)
  - app-linear: same with preprocess=False (the bare MDF filter; what the
                pip binding gives you)
  - swig:       the pip `speexdsp` binding (cross-check that our ctypes shim
                matches it; needs a locally repaired build - the 2018 PyPI
                sdist ships a Python-2-era SWIG proxy that breaks on 3.12)

Usage: .venv/bin/python spikes/aec_probe.py [--tail-ms 400] [--chunk 480]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sphn
from scipy.signal import butter, lfilter

try:
    from moneypenny.aec import EchoCanceller as AppEchoCanceller
except Exception:  # recon phase: module may not exist yet
    AppEchoCanceller = None

try:
    from speexdsp import EchoCanceller as SwigEchoCanceller
except Exception:  # broken upstream packaging on py3.12 unless locally repaired
    SwigEchoCanceller = None

SAMPLE_RATE = 24000
FRAME = 1920
OUT = Path(__file__).parent / "out"

NEAR_ONSET_S = 8.0       # double-talk starts here
ECHO_GAIN_RMS = 0.05     # echo level at the mic (open speakers near the mic)
NOISE_RMS = 0.002        # room tone
WINDOW_S = 0.5
MIN_ECHO_RMS = 0.01      # windows quieter than this (far-signal pauses) are skipped


def load_fixture(name: str) -> np.ndarray:
    pcm, _ = sphn.read(str(OUT / name), sample_rate=SAMPLE_RATE)
    return pcm[0].astype(np.float32)


def synth_echo(far: np.ndarray) -> np.ndarray:
    """Linear room/speaker model: 30ms bulk delay with early-reflection smear,
    a 47ms second bounce, then a 2nd-order 7kHz rolloff (speaker response)."""
    h = np.zeros(2400, dtype=np.float32)
    h[720:726] = [0.32, 0.22, 0.15, 0.09, 0.05, 0.03]
    h[1128:1132] = [0.11, 0.06, 0.04, 0.02]
    echo = np.convolve(far, h)[: len(far)]
    b, a = butter(2, 7000, fs=SAMPLE_RATE)
    echo = lfilter(b, a, echo).astype(np.float32)
    return echo * (ECHO_GAIN_RMS / float(np.sqrt(np.mean(echo**2))))


def build_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mic, far, near_track), all the same length."""
    far = np.tile(load_fixture("briefing.wav"), 3)  # ~13.7s of model speech
    near = load_fixture("question.wav")             # ~2.7s of user speech
    near_track = np.zeros_like(far)
    onset = int(NEAR_ONSET_S * SAMPLE_RATE)
    near_track[onset : onset + len(near)] = near
    rng = np.random.default_rng(0xAEC)
    noise = rng.normal(0.0, NOISE_RMS, len(far)).astype(np.float32)
    mic = synth_echo(far) + near_track + noise
    return mic.astype(np.float32), far, near_track


class SwigAdapter:
    """float32 frames -> the pip binding's int16-bytes process(mic, ref)."""

    def __init__(self, chunk: int, tail_ms: int) -> None:
        tail = int(SAMPLE_RATE * tail_ms / 1000)
        self._ec = SwigEchoCanceller.create(chunk, tail, SAMPLE_RATE)
        self._chunk = chunk

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        out = np.empty_like(mic)
        for i in range(0, len(mic), self._chunk):
            m = (np.clip(mic[i : i + self._chunk], -1, 1) * 32767).astype("<i2")
            r = (np.clip(ref[i : i + self._chunk], -1, 1) * 32767).astype("<i2")
            res = self._ec.process(m.tobytes(), r.tobytes())
            out[i : i + self._chunk] = (
                np.frombuffer(res, dtype="<i2").astype(np.float32) / 32767.0
            )
        return out


def run(canceller, mic: np.ndarray, far: np.ndarray) -> tuple[np.ndarray, list[float]]:
    out = np.zeros_like(mic)
    times: list[float] = []
    for i in range(0, len(mic) - FRAME + 1, FRAME):
        t0 = time.perf_counter()
        out[i : i + FRAME] = canceller.process(mic[i : i + FRAME], far[i : i + FRAME])
        times.append(time.perf_counter() - t0)
    return out, times


def lag_align(out: np.ndarray, ref: np.ndarray, max_lag: int = 480) -> tuple[float, np.ndarray, np.ndarray]:
    """(peak corr, aligned out, aligned ref) over +-max_lag samples, 1ms steps.
    The preprocessor's FFT block delay must read as delay, not as damage."""
    best, best_pair = -1.0, (out, ref)
    for lag in range(-max_lag, max_lag + 1, 24):
        if lag >= 0:
            a, b = out[lag:], ref[: len(ref) - lag]
        else:
            a, b = out[: len(out) + lag], ref[-lag:]
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best:
            best, best_pair = c, (a, b)
    return best, *best_pair


def windowed_erle(mic: np.ndarray, out: np.ndarray, lo_s: float, hi_s: float) -> list[tuple[float, float]]:
    """[(window_start_s, erle_db)] over echo-active windows in [lo_s, hi_s)."""
    win = int(WINDOW_S * SAMPLE_RATE)
    rows = []
    for start in range(int(lo_s * SAMPLE_RATE), int(hi_s * SAMPLE_RATE) - win + 1, win):
        m = mic[start : start + win]
        if float(np.sqrt(np.mean(m**2))) < MIN_ECHO_RMS:
            continue  # far-signal pause: nothing to cancel here
        o = out[start : start + win]
        erle = 10 * np.log10(np.mean(m**2) / max(np.mean(o**2), 1e-12))
        rows.append((start / SAMPLE_RATE, float(erle)))
    return rows


def evaluate(name: str, make_canceller) -> dict:
    mic, far, near_track = build_scene()
    out, times = run(make_canceller(), mic, far)

    convergence = windowed_erle(mic, out, 0.0, NEAR_ONSET_S)
    to_10db = next((t for t, e in convergence if e > 10.0), None)
    settled = [e for t, e in convergence if t >= 6.0]
    converged_erle = float(np.mean(settled)) if settled else float("nan")

    onset = int(NEAR_ONSET_S * SAMPLE_RATE)
    span = slice(onset, onset + int(2.7 * SAMPLE_RATE))
    near_corr, out_al, near_al = lag_align(out[span], near_track[span])
    # SDR on lag-aligned, gain-matched signals: how much non-near junk
    # (residual echo, suppression artifacts) rides along with user speech.
    gain = float(np.dot(out_al, near_al) / max(np.dot(near_al, near_al), 1e-12))
    near_sdr = 10 * np.log10(
        np.mean(near_al**2) / max(np.mean((out_al / max(gain, 1e-6) - near_al) ** 2), 1e-12)
    )

    ms = np.array(times) * 1000
    result = {
        "impl": name,
        "erle_to_10db_s": to_10db,
        "converged_erle_db": round(converged_erle, 1),
        "double_talk_near_corr": round(near_corr, 3),
        "double_talk_near_sdr_db": round(float(near_sdr), 1),
        "frame_ms_mean": round(float(ms.mean()), 3),
        "frame_ms_p99": round(float(np.percentile(ms, 99)), 3),
        "frame_ms_max": round(float(ms.max()), 3),
        "convergence_curve": [(round(t, 1), round(e, 1)) for t, e in convergence],
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail-ms", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=480)
    args = ap.parse_args()

    impls = {}
    if AppEchoCanceller is not None:
        impls["app"] = lambda: AppEchoCanceller(
            frame_samples=FRAME, sample_rate=SAMPLE_RATE,
            filter_ms=args.tail_ms, chunk_samples=args.chunk,
        )
        impls["app-linear"] = lambda: AppEchoCanceller(
            frame_samples=FRAME, sample_rate=SAMPLE_RATE,
            filter_ms=args.tail_ms, chunk_samples=args.chunk, preprocess=False,
        )
    if SwigEchoCanceller is not None:
        impls["swig"] = lambda: SwigAdapter(args.chunk, args.tail_ms)
    if not impls:
        raise SystemExit("no echo canceller importable (moneypenny.aec or speexdsp)")

    results = []
    for name, make in impls.items():
        r = evaluate(f"{name} tail={args.tail_ms}ms chunk={args.chunk}", make)
        results.append(r)
        curve = " ".join(f"{t}s:{e}dB" for t, e in r.pop("convergence_curve")[:8])
        print(json.dumps(r, indent=2))
        print(f"  early convergence: {curve}")

    (OUT / "aec_probe.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / 'aec_probe.json'}")


if __name__ == "__main__":
    main()
