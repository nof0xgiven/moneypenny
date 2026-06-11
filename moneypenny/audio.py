"""Duplex audio I/O. Mirrors personaplex_mlx/local.py's client (proven pattern):
callbacks push/pull 1920-sample frames through queues. No processing here."""
from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24000
FRAME = 1920


class AudioIO:
    def __init__(self) -> None:
        self.mic_frames: "queue.Queue[np.ndarray]" = queue.Queue()
        self.speaker_frames: "queue.Queue[np.ndarray]" = queue.Queue()
        self._in = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME, callback=self._on_input
        )
        self._out = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME, callback=self._on_output
        )

    def _on_input(self, in_data, frames, timing, status) -> None:
        self.mic_frames.put_nowait(in_data[:, 0].astype(np.float32).copy())

    def _on_output(self, out_data, frames, timing, status) -> None:
        try:
            out_data[:, 0] = self.speaker_frames.get_nowait()
        except queue.Empty:
            out_data.fill(0)

    def __enter__(self) -> "AudioIO":
        self._in.start()
        self._out.start()
        return self

    def __exit__(self, *exc) -> None:
        # stop-then-close per stream; nested finally so a raise anywhere can
        # never skip closing the other PortAudio stream.
        try:
            try:
                self._in.stop()
            finally:
                self._in.close()
        finally:
            try:
                self._out.stop()
            finally:
                self._out.close()
