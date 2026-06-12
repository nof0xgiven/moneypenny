"""Echo-canceller tests: real speexdsp DSP on synthetic echo paths, no mocks.

The fast tests use deterministic speech-like signals (harmonic stack + AM +
breath noise) so they run anywhere without the gitignored TTS fixtures; the
slow test replays the spikes/aec_probe.py scenario on the real fixture WAVs.
Thresholds carry margin below the probe's measured numbers (linear ERLE
~17dB, +preprocess ~34dB, near-speech correlation 0.96, ~0.4ms/frame):
they catch a broken canceller, not a 1dB regression.

AudioIO wiring is tested by calling the PortAudio callbacks directly as the
plain methods they are -- no devices are opened (streams are created in
__enter__, which these tests never call).
"""
from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import butter, lfilter

from moneypenny.aec import EchoCanceller
from moneypenny.audio import FRAME, SAMPLE_RATE, AudioIO

ECHO_DELAY = 720  # 30ms device+room latency
ECHO_GAIN = 0.55


def speechish(n: int, seed: int, f0: float = 120.0) -> np.ndarray:
    """Deterministic speech-like signal: harmonic stack with syllable-rate AM
    plus a little breath noise (broadband content keeps MDF adaptation fed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SAMPLE_RATE
    sig = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi)) / h
              for h in range(1, 9))
    am = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi)))
    sig = sig * am + 0.05 * rng.standard_normal(n)
    return (0.1 * sig / np.max(np.abs(sig))).astype(np.float32)


def echo_of(played: np.ndarray) -> np.ndarray:
    """The synthetic room: delay + attenuate + lowpass (speaker coloration)."""
    echoed = np.zeros_like(played)
    echoed[ECHO_DELAY:] = played[:-ECHO_DELAY] * ECHO_GAIN
    b, a = butter(4, 4000.0 / (SAMPLE_RATE / 2))
    return lfilter(b, a, echoed).astype(np.float32)


def _db(p_in: float, p_out: float) -> float:
    return float(10.0 * np.log10((p_in + 1e-12) / (p_out + 1e-12)))


def _lag_corr(out: np.ndarray, ref: np.ndarray, max_lag: int = 480) -> float:
    """Normalized correlation peak over +-max_lag samples (the preprocessor's
    FFT block delay must not read as damage)."""
    best = -1.0
    for lag in range(-max_lag, max_lag + 1, 24):
        if lag >= 0:
            a, b = out[lag:], ref[:len(ref) - lag]
        else:
            a, b = out[:len(out) + lag], ref[-lag:]
        best = max(best, float(np.corrcoef(a, b)[0, 1]))
    return best


def run_frames(aec: EchoCanceller, mic: np.ndarray, far: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mic)
    for i in range(len(mic) // FRAME):
        s = slice(i * FRAME, (i + 1) * FRAME)
        out[s] = aec.process(mic[s], far[s])
    return out


def test_converges_on_synthetic_echo_path():
    n = 6 * SAMPLE_RATE
    far = speechish(n, seed=1, f0=120.0)
    mic = echo_of(far) + np.random.default_rng(9).standard_normal(n).astype(np.float32) * 0.001
    out = run_frames(EchoCanceller(), mic, far)

    tail = slice(4 * SAMPLE_RATE, n)  # converged region
    erle = _db(np.mean(mic[tail] ** 2), np.mean(out[tail] ** 2))
    assert erle > 10.0, f"converged ERLE {erle:.1f}dB"

    # convergence: some 250ms window inside the first 3s already exceeds 10dB
    win = SAMPLE_RATE // 4
    sliding = [
        _db(np.mean(mic[s:s + win] ** 2), np.mean(out[s:s + win] ** 2))
        for s in range(0, 3 * SAMPLE_RATE - win, win // 2)
    ]
    assert max(sliding) > 10.0, f"never converged within 3s: max {max(sliding):.1f}dB"


def test_double_talk_preserves_near_speech():
    n = 8 * SAMPLE_RATE
    far = speechish(n, seed=2, f0=120.0)
    near = np.zeros(n, dtype=np.float32)
    near[5 * SAMPLE_RATE:] = speechish(3 * SAMPLE_RATE, seed=3, f0=185.0)
    mic = echo_of(far) + near
    out = run_frames(EchoCanceller(), mic, far)

    dt = slice(5 * SAMPLE_RATE, n)
    corr = _lag_corr(out[dt], near[dt])
    assert corr > 0.85, f"near speech mangled in double-talk: corr {corr:.3f}"


def test_model_silent_passthrough():
    """Far silent (ref all zeros): user speech must come through unscathed."""
    n = 4 * SAMPLE_RATE
    near = speechish(n, seed=4, f0=185.0)
    out = run_frames(EchoCanceller(), near, np.zeros(n, dtype=np.float32))

    tail = slice(SAMPLE_RATE, n)
    corr = _lag_corr(out[tail], near[tail])
    drop = _db(np.mean(near[tail] ** 2), np.mean(out[tail] ** 2))
    assert corr > 0.9, f"silent-ref passthrough mangled speech: corr {corr:.3f}"
    assert drop < 3.0, f"silent-ref passthrough lost {drop:.1f}dB of user speech"


def test_per_frame_cost_within_callback_budget():
    """The canceller runs on the PortAudio input callback thread: its cost per
    80ms frame must stay invisible (<2ms; measured ~0.4ms). If this ever
    fails, AEC must move off the callback into the frame loop."""
    import time

    aec = EchoCanceller()
    far = speechish(100 * FRAME, seed=5)
    mic = echo_of(far)
    t0 = time.perf_counter()
    run_frames(aec, mic, far)
    per_frame_ms = (time.perf_counter() - t0) / 100 * 1000
    assert per_frame_ms < 2.0, f"{per_frame_ms:.2f}ms per frame"


def test_rejects_frame_not_divisible_by_chunk():
    with pytest.raises(ValueError):
        EchoCanceller(frame_samples=1000, chunk_samples=480)


# --- AudioIO wiring (callbacks driven directly; no devices opened) ---


def _in_buf(frame: np.ndarray) -> np.ndarray:
    return frame.reshape(-1, 1)


def _drain(q: "queue.Queue[np.ndarray]") -> list[np.ndarray]:
    frames = []
    while True:
        try:
            frames.append(q.get_nowait())
        except queue.Empty:
            return frames


def test_audioio_cancels_speaker_echo_on_mic_path():
    n_frames = 75  # 6s
    far = speechish(n_frames * FRAME, seed=6)
    echo = echo_of(far)
    audio = AudioIO(aec=EchoCanceller())
    out_buf = np.zeros((FRAME, 1), dtype=np.float32)
    for i in range(n_frames):
        s = slice(i * FRAME, (i + 1) * FRAME)
        audio.speaker_frames.put_nowait(far[s])
        audio._on_output(out_buf, FRAME, None, None)
        audio._on_input(_in_buf(echo[s]), FRAME, None, None)

    got = np.concatenate(_drain(audio.mic_frames))
    tail = slice(4 * SAMPLE_RATE, n_frames * FRAME)
    erle = _db(np.mean(echo[tail] ** 2), np.mean(got[tail] ** 2))
    assert erle > 10.0, f"wired ERLE {erle:.1f}dB"


def test_audioio_pairing_survives_bursty_callback_order():
    """Live failure mode (measured 2026-06-12): MLX workers hold the GIL in
    ~60ms chunks, so the PortAudio callbacks fire in BURSTS (out,out,in,in,
    out,in,...) instead of strict alternation. Arrival-order pairing then
    slips a frame on every burst and the filter chases a wandering bulk
    delay (live xcorr lag jumped 2ms<->110ms<->387ms; no convergence, echo
    passed through at full level). Pairing must key on stream position --
    the k-th mic frame against the k-th played frame -- which burst order
    cannot disturb: each callback still carries the next block of its
    stream."""
    n_frames = 76  # multiple of the 4-callback burst pattern
    far = speechish(n_frames * FRAME, seed=11)
    echo = echo_of(far)
    audio = AudioIO(aec=EchoCanceller())
    out_buf = np.zeros((FRAME, 1), dtype=np.float32)

    def out_cb(i: int) -> None:
        audio.speaker_frames.put_nowait(far[i * FRAME:(i + 1) * FRAME])
        audio._on_output(out_buf, FRAME, None, None)

    def in_cb(i: int) -> None:
        audio._on_input(_in_buf(echo[i * FRAME:(i + 1) * FRAME]), FRAME, None, None)

    oi = ii = 0
    while oi < n_frames or ii < n_frames:
        # burst pattern: two output callbacks land, then two inputs catch up
        for _ in range(2):
            if oi < n_frames:
                out_cb(oi)
                oi += 1
        for _ in range(2):
            if ii < n_frames:
                in_cb(ii)
                ii += 1

    got = np.concatenate(_drain(audio.mic_frames))
    tail = slice(4 * SAMPLE_RATE, n_frames * FRAME)
    erle = _db(np.mean(echo[tail] ** 2), np.mean(got[tail] ** 2))
    assert erle > 10.0, f"bursty-order ERLE {erle:.1f}dB (pairing slipped)"


def test_audioio_underrun_keeps_ref_stream_aligned():
    """Speaker queue empty -> the callback played zeros -> the ref ring must
    receive zeros too (stream continuity), so user speech while the model is
    silent passes through uncancelled."""
    audio = AudioIO(aec=EchoCanceller())
    out_buf = np.zeros((FRAME, 1), dtype=np.float32)
    near = speechish(40 * FRAME, seed=7, f0=185.0)
    for i in range(40):  # all underruns: model never speaks
        audio._on_output(out_buf, FRAME, None, None)
        s = slice(i * FRAME, (i + 1) * FRAME)
        audio._on_input(_in_buf(near[s]), FRAME, None, None)

    assert audio.underruns == 40
    got = np.concatenate(_drain(audio.mic_frames))
    tail = slice(10 * FRAME, 40 * FRAME)
    corr = _lag_corr(got[tail], near[tail])
    assert corr > 0.9, f"underrun-ref passthrough mangled speech: corr {corr:.3f}"


def test_audioio_without_aec_is_bit_identical_passthrough():
    audio = AudioIO()
    frame = speechish(FRAME, seed=8)
    audio._on_input(_in_buf(frame), FRAME, None, None)
    got = audio.mic_frames.get_nowait()
    np.testing.assert_array_equal(got, frame)


def test_audioio_ref_ring_is_bounded():
    """Output callbacks without matching input callbacks (input stream stall)
    must not grow the ref ring without bound."""
    audio = AudioIO(aec=EchoCanceller())
    out_buf = np.zeros((FRAME, 1), dtype=np.float32)
    for _ in range(100):
        audio.speaker_frames.put_nowait(np.zeros(FRAME, dtype=np.float32))
        audio._on_output(out_buf, FRAME, None, None)
    assert len(audio._ref_ring) <= 8


def test_audioio_reanchors_after_mic_side_sample_loss():
    """The live failure mode (tmp dump, 2026-06-12 session): during an engine
    warm-up CPU spike CoreAudio silently dropped ~2 frames of MIC samples (no
    overflow flag) while the output stream kept playing. Blind positional
    pairing then hands the canceller reference frames that played BEFORE the
    mic frame's audio was captured -- acausal, so the MDF filter can never
    cancel again (measured live: ~0-2dB suppression for the rest of the
    session). With dac/adc timestamp anchoring, the stale reference frames
    must be dropped (counted as slips) and ERLE must recover after the slip.

    For the ERLE assertion to DISCRIMINATE broken pairing from fixed, two
    confounds must be designed out (review finding: with a speechish far
    signal even blind FIFO pairing scored ~30dB here):
      - the far signal must be aperiodic AND nonstationary -- a pitch-
        periodic fixture self-aligns at the period, and a stationary noise
        fixture gets eaten by the preprocessor's denoiser either way;
      - the canceller runs preprocess=False -- the preprocessor suppresses
        by spectral gain, which needs no waveform alignment and masks ~14dB
        regardless of pairing. The linear MDF filter is the component whose
        causality the anchoring protects, so it is what this test measures.
    Measured on this scenario: blind FIFO 0.1dB (fails), anchored 29dB.

    Timestamps mirror the measured hardware: both PortAudio streams share one
    host-clock epoch; out latency ~25ms, in latency ~12ms.
    """
    n_frames = 75
    # aperiodic + nonstationary: noise carrier under a deep random slow
    # envelope; no period to self-align on, no stationary floor to denoise
    rng = np.random.default_rng(10)
    carrier = rng.standard_normal(n_frames * FRAME)
    b, a = butter(2, 2.5 / (SAMPLE_RATE / 2))
    env = lfilter(b, a, np.abs(rng.standard_normal(n_frames * FRAME)))
    env -= env.min()
    env /= env.max()
    far = (0.15 * carrier * env ** 2).astype(np.float32)
    echo = echo_of(far)
    # the mic never digitized frames 30-31: splice them OUT of the echo
    # stream (content shifts earlier; this is sample loss, not silence)
    cut = np.concatenate([echo[:30 * FRAME], echo[32 * FRAME:]])
    n_in = len(cut) // FRAME

    audio = AudioIO(aec=EchoCanceller(preprocess=False))
    out_buf = np.zeros((FRAME, 1), dtype=np.float32)
    frame_s = FRAME / SAMPLE_RATE

    def out_t(k: int):  # k-th output callback: frame hits the DAC 25ms later
        return SimpleNamespace(outputBufferDacTime=1000.0 + k * frame_s + 0.025)

    def in_t(j: int):  # j-th input callback: capture of 80ms ending 12ms ago
        k = j if j < 30 else j + 2  # post-loss content was captured 2 frames later
        return SimpleNamespace(inputBufferAdcTime=1000.0 + k * frame_s - 0.012 - frame_s)

    j = 0
    for k in range(n_frames):
        audio.speaker_frames.put_nowait(far[k * FRAME:(k + 1) * FRAME])
        audio._on_output(out_buf, FRAME, out_t(k), None)
        # input misses its tick twice while the output keeps going (the stall)
        if k in (30, 31):
            continue
        if j < n_in:
            audio._on_input(_in_buf(cut[j * FRAME:(j + 1) * FRAME]), FRAME, in_t(j), None)
            j += 1

    got = np.concatenate(_drain(audio.mic_frames))
    # well after the slip (>=1.8s of re-convergence): echo must be cancelled again
    tail = slice(53 * FRAME, n_in * FRAME)
    erle = _db(np.mean(cut[tail] ** 2), np.mean(got[tail] ** 2))
    assert erle > 10.0, f"post-slip ERLE {erle:.1f}dB (canceller never re-anchored)"
    assert audio.ref_slips == 2, f"expected 2 stale refs dropped, got {audio.ref_slips}"


# --- real-fixture replay (the probe scenario, real TTS speech) ---


@pytest.mark.slow
def test_real_speech_fixture_convergence():
    sphn = pytest.importorskip("sphn")
    try:
        far_src, _ = sphn.read("spikes/out/question.wav", sample_rate=SAMPLE_RATE)
        near_src, _ = sphn.read("spikes/out/briefing_male.wav", sample_rate=SAMPLE_RATE)
    except Exception:
        pytest.skip("fixtures missing: run spikes/make_fixtures.py")
    n = 8 * SAMPLE_RATE
    far = np.tile(far_src[0], int(np.ceil(n / far_src.shape[-1])))[:n].astype(np.float32)
    near = np.zeros(n, dtype=np.float32)
    seg = near_src[0][: 2 * SAMPLE_RATE].astype(np.float32)
    near[6 * SAMPLE_RATE:6 * SAMPLE_RATE + len(seg)] = seg
    mic = echo_of(far) + near
    out = run_frames(EchoCanceller(), mic, far)

    tail = slice(4 * SAMPLE_RATE, 6 * SAMPLE_RATE)
    erle = _db(np.mean(mic[tail] ** 2), np.mean(out[tail] ** 2))
    dt = slice(6 * SAMPLE_RATE, n)
    corr = _lag_corr(out[dt], near[dt])
    assert erle > 10.0, f"real-speech ERLE {erle:.1f}dB"
    assert corr > 0.85, f"real-speech double-talk corr {corr:.3f}"
