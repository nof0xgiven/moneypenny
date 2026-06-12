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
        # AEC state (None = pass-through). AudioIO owns the canceller once
        # handed in: __exit__ closes it after the streams stop.
        self._aec = aec
        self._ref_ring: "collections.deque[np.ndarray]" = collections.deque(maxlen=REF_RING_FRAMES)
        self._ref_lock = threading.Lock()
        self._ref_silence = np.zeros(FRAME, dtype=np.float32)
        self._in: sd.InputStream | None = None
        self._out: sd.OutputStream | None = None

    def _on_input(self, in_data, frames, timing, status) -> None:
        mic = in_data[:, 0].astype(np.float32).copy()
        if self._aec is not None:
            with self._ref_lock:
                ref = self._ref_ring.popleft() if self._ref_ring else self._ref_silence
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
            played = out_data[:, 0].astype(np.float32).copy()
            with self._ref_lock:
                self._ref_ring.append(played)

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
