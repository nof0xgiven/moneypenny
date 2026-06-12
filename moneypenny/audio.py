"""Duplex audio I/O. Mirrors personaplex_mlx/local.py's client (proven pattern):
callbacks push/pull 1920-sample frames through queues.

The one piece of processing that lives here is echo cancellation, because it
is intrinsically a device-path concern: the output callback is the only place
that knows exactly which PCM the speakers played (including underrun
zero-fill), and that exact stream is the AEC's far-end reference. The
reference crosses from the output-callback thread to the input-callback
thread through a small lock-guarded FIFO ring:

  _on_output  appends a copy of every frame it actually played (deque
              maxlen=8 absorbs callback jitter; both streams tick at the
              same blocksize, so steady-state depth is ~1)
  _on_input   pops exactly one reference frame per mic frame (zeros when the
              ring is empty: output stream not started yet, or stalled) and
              runs the canceller BEFORE the frame enters mic_frames

FIFO, not latest-frame-wins: the speex MDF filter models the room as a
convolution of the continuous far-end stream; every played sample must reach
process() exactly once, in order (moneypenny/aec.py docstring). The deque
only drops (oldest first) if the input stream stalls outright, which already
desyncs the streams more than the drop does; the 400ms filter re-converges.

Timestamp anchoring on top of the FIFO: blind positional pairing assumes
neither hardware stream ever loses samples. Measured live (2026-06-12, RØDE
+ engine warm-up CPU spike), CoreAudio dropped ~2 frames of MIC samples with
NO overflow flag while the output kept playing; from then on every reference
frame the FIFO handed the canceller had played BEFORE the paired mic frame
was captured -- acausal, unrecoverable, echo passed through for the rest of
the session. PortAudio's hardware timestamps expose the loss: the dac->adc
skew of a healthy pairing is constant (callback EXECUTION may run late under
MLX GIL bursts, but hardware time does not lie), so a jump beyond half a
frame marks a real stream slip. The pairing then drops stale reference
frames (counted in ref_slips) until the skew re-anchors, and hands zeros
when the reference is in the mic frame's future (DAC gap: zeros really were
played). Callbacks driven without timestamps (unit tests, hosts that supply
none) fall back to the plain FIFO.

AEC cost on the input-callback thread is ~0.4ms per 80ms frame (measured:
tests/test_aec.py::test_per_frame_cost_within_callback_budget), invisible
next to the budget. Streams are opened in __enter__, not __init__, so the
callbacks stay plain testable methods and a constructed-but-unentered
AudioIO never touches PortAudio.
"""
from __future__ import annotations

import collections
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from moneypenny.aec import EchoCanceller

SAMPLE_RATE = 24000
FRAME = 1920
REF_RING_FRAMES = 8
# Skew deviation beyond this marks a genuine stream slip (real hardware
# timestamps jitter by ~ms; sample loss shifts them by whole frames).
SLIP_THRESHOLD_S = 0.04


class AudioIO:
    def __init__(self, aec: "EchoCanceller | None" = None) -> None:
        self.mic_frames: "queue.Queue[np.ndarray]" = queue.Queue()
        self.speaker_frames: "queue.Queue[np.ndarray]" = queue.Queue()
        # Output underrun diagnostic: total zero-filled callbacks since start.
        # Bursts of growth while the model is mid-sentence are the direct
        # evidence of broken (choppy) speaker output. Written only by the
        # PortAudio output-callback thread (so the += read-modify-write never
        # races another writer); the status line on the loop thread only
        # reads, and a momentarily stale read is fine for a diagnostic.
        self.underruns = 0
        # Reference-pairing slips: stale reference frames dropped after a
        # hardware stream lost samples (see module docstring). A burst of
        # growth marks a slip event; the canceller re-converges after it.
        self.ref_slips = 0
        # AEC state (None = pass-through). AudioIO owns the canceller once
        # handed in: __exit__ closes it after the streams stop.
        self._aec = aec
        # ring of (dac_time | None, played frame); dac_time None when the
        # host supplies no timestamp (plain-FIFO fallback pairing)
        self._ref_ring: "collections.deque[tuple[float | None, np.ndarray]]" = (
            collections.deque(maxlen=REF_RING_FRAMES)
        )
        self._ref_lock = threading.Lock()
        self._ref_skew: float | None = None  # anchored adc-dac of a healthy pairing
        self._ref_silence = np.zeros(FRAME, dtype=np.float32)
        self._in: sd.InputStream | None = None
        self._out: sd.OutputStream | None = None

    def _pop_ref(self, adc: float) -> np.ndarray:
        """Choose this mic frame's far-end reference. Caller holds _ref_lock."""
        if not self._ref_ring:
            return self._ref_silence
        dac = self._ref_ring[0][0]
        if not adc or dac is None:
            return self._ref_ring.popleft()[1]  # no timestamps: positional FIFO
        if self._ref_skew is None:
            self._ref_skew = adc - dac  # anchor on the first timestamped pairing
        # drop references that played too long before this mic audio was
        # captured (the mic stream lost samples; pairing them is acausal)
        while self._ref_ring and self._ref_ring[0][0] is not None \
                and adc - self._ref_ring[0][0] > self._ref_skew + SLIP_THRESHOLD_S:
            self._ref_ring.popleft()
            self.ref_slips += 1
        if not self._ref_ring:
            return self._ref_silence
        dac = self._ref_ring[0][0]
        if dac is None or abs((adc - dac) - self._ref_skew) <= SLIP_THRESHOLD_S:
            return self._ref_ring.popleft()[1]
        # head is in this mic frame's future. Normally a transient (DAC gap:
        # zeros really were played; keep the frame for its later mic frame),
        # but a full ring means the output timeline itself jumped -- re-anchor
        # on the head and take the re-convergence hit.
        if len(self._ref_ring) == self._ref_ring.maxlen:
            self._ref_skew = adc - dac
            self.ref_slips += 1
            return self._ref_ring.popleft()[1]
        return self._ref_silence

    def _on_input(self, in_data, frames, timing, status) -> None:
        mic = in_data[:, 0].astype(np.float32).copy()
        if self._aec is not None:
            adc = float(getattr(timing, "inputBufferAdcTime", 0.0) or 0.0) if timing else 0.0
            with self._ref_lock:
                ref = self._pop_ref(adc)
            mic = self._aec.process(mic, ref)
        self.mic_frames.put_nowait(mic)

    def _on_output(self, out_data, frames, timing, status) -> None:
        try:
            out_data[:, 0] = self.speaker_frames.get_nowait()
        except queue.Empty:
            self.underruns += 1
            out_data.fill(0)
        if self._aec is not None:
            # what was ACTUALLY played -- real frame or underrun zeros -- is
            # the far-end reference; both must enter the stream (aec.py).
            dac = float(getattr(timing, "outputBufferDacTime", 0.0) or 0.0) if timing else 0.0
            played = out_data[:, 0].astype(np.float32).copy()
            with self._ref_lock:
                self._ref_ring.append((dac if dac else None, played))

    def __enter__(self) -> "AudioIO":
        # latency="low": sounddevice's default ("high") let CoreAudio buffer
        # the RØDE input by ~850ms (measured), which (a) delays every mic
        # frame the conversation sees by close to a second and (b) pushes the
        # speaker->mic echo ~1s past its AEC reference frame -- no realistic
        # filter length covers that. Low latency measured ~12ms in / ~25ms
        # out on this hardware; the frame queues remain the jitter buffers.
        self._in = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME, latency="low",
            callback=self._on_input,
        )
        try:
            self._out = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME, latency="low",
                callback=self._on_output,
            )
            try:
                self._in.start()
                self._out.start()
            except BaseException:
                self._out.close(ignore_errors=True)
                raise
        except BaseException:
            # a failed enter leaves nothing open (__exit__ never runs when
            # __enter__ raises)
            self._in.close(ignore_errors=True)
            self._in = self._out = None
            raise
        return self

    def __exit__(self, *exc) -> None:
        # stop-then-close per stream; nested finally so a raise anywhere can
        # never skip closing the other PortAudio stream or the AEC state.
        try:
            try:
                if self._in is not None:
                    self._in.stop()
            finally:
                if self._in is not None:
                    self._in.close()
        finally:
            try:
                try:
                    if self._out is not None:
                        self._out.stop()
                finally:
                    if self._out is not None:
                        self._out.close()
            finally:
                if self._aec is not None:
                    self._aec.close()
