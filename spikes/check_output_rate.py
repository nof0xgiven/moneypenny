"""H-C diagnostic: does OutputStream(24000) on the default device play at the
right speed? Play 2s of 440Hz sine through the room speakers while recording
the room mic; FFT the capture. ~440Hz peak = PortAudio resamples correctly;
~220Hz = half-speed playback (sample-rate mismatch). Also wall-clocks the
blocking writes: 2s of audio should take ~2s to write."""
from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

OUT_SR = 24000
REC_SR = 48000
TONE_HZ = 440.0
DUR_S = 2.0

print("default devices:", sd.default.device)
for kind in ("input", "output"):
    d = sd.query_devices(kind=kind)
    print(f"{kind}: {d['name']!r} default_sr={d['default_samplerate']:.0f} "
          f"max_{kind}_channels={d[f'max_{kind}_channels']}")

t = np.arange(int(OUT_SR * DUR_S)) / OUT_SR
tone = (0.4 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)

rec_frames: list[np.ndarray] = []


def on_rec(indata, frames, timing, status):
    rec_frames.append(indata[:, 0].copy())


with sd.InputStream(samplerate=REC_SR, channels=1, callback=on_rec):
    out = sd.OutputStream(samplerate=OUT_SR, channels=1)
    out.start()
    print(f"opened OutputStream: samplerate={out.samplerate} active={out.active}")
    t0 = time.perf_counter()
    # blocking writes in 80ms chunks, paced by PortAudio's own backpressure
    chunk = 1920
    for i in range(0, len(tone), chunk):
        out.write(tone[i:i + chunk].reshape(-1, 1))
    out.stop()
    wall = time.perf_counter() - t0
    out.close()
    sd.sleep(300)  # let the mic capture the tail

rec = np.concatenate(rec_frames)
print(f"wrote {DUR_S}s of audio in {wall:.2f}s wall-clock "
      f"({'OK' if abs(wall - DUR_S) < 0.4 else 'ANOMALOUS'})")

# FFT the middle of the capture (skip start/stop transients)
mid = rec[len(rec) // 4: 3 * len(rec) // 4]
win = np.hanning(len(mid))
spec = np.abs(np.fft.rfft(mid * win))
freqs = np.fft.rfftfreq(len(mid), 1 / REC_SR)
band = (freqs > 100) & (freqs < 1000)
peak_hz = freqs[band][np.argmax(spec[band])]
print(f"captured {len(rec)/REC_SR:.2f}s; dominant peak in 100-1000Hz: {peak_hz:.1f} Hz")
if abs(peak_hz - TONE_HZ) < 20:
    print("VERDICT: ~440Hz -> output sample rate handled correctly (H-C rejected)")
elif abs(peak_hz - TONE_HZ / 2) < 20:
    print("VERDICT: ~220Hz -> HALF-SPEED playback (sample-rate mismatch, H-C confirmed)")
else:
    print("VERDICT: unexpected peak; inspect manually")
