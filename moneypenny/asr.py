"""Streaming ASR over parakeet-mlx. Input: 24kHz frames (Mimi-native); the
model wants 16kHz, so we resample per frame (ratio 2/3, polyphase)."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained
from scipy.signal import resample_poly


def resample_24k_to_16k(pcm: np.ndarray) -> np.ndarray:
    return resample_poly(pcm.astype(np.float32), up=2, down=3).astype(np.float32)


class StreamingTranscriber:
    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v3") -> None:
        self._model = from_pretrained(model_id)
        self._ctx = self._model.transcribe_stream(context_size=(256, 256))
        self._ctx.__enter__()

    def add_frame(self, pcm_24k: np.ndarray) -> str:
        """Feed one 24kHz frame; returns current partial transcript ('' if none)."""
        self._ctx.add_audio(mx.array(resample_24k_to_16k(pcm_24k)))
        return self._ctx.result.text

    def finish(self) -> str:
        text = self._ctx.result.text
        self._ctx.__exit__(None, None, None)
        return text

    def reset(self) -> None:
        """Start a fresh utterance context."""
        try:
            self._ctx.__exit__(None, None, None)
        except AttributeError:
            # Already exited (e.g. after finish()): the context's buffers were
            # deleted, so a second __exit__ raises AttributeError.
            pass
        self._ctx = self._model.transcribe_stream(context_size=(256, 256))
        self._ctx.__enter__()
