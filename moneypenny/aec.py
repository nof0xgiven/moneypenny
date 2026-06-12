"""Acoustic echo cancellation for the mic path (speexdsp, bound via ctypes).

Why this exists: on open speakers the model's own voice leaks into the mic,
trips the VAD, and ASR transcribes phantom "user" utterances of the model's
words (the classify gate catches the clean ones; garbled and tool-keyword
phantoms slip through). We hold a structural advantage over generic AEC
deployments: every PCM frame the speakers play passes through our own output
callback, so the far-end reference is exact, not estimated.

Why ctypes and not the pip bindings (recon evidence: spikes/aec_probe.py):

- pip `speexdsp` 0.1.1 compiles on macOS arm64 but ships a pre-generated
  SWIG 2.0.11 wrapper that does `import imp` -- removed in Python 3.12, so
  the package is dead at import time. Its extension dynamically links brew's
  libspeexdsp anyway, and it never wrapped the preprocessor stage.
- pip `webrtc-audio-processing` 0.1.3 hardcodes ARM32 compiler flags
  (-mfloat-abi=/-mfpu=); clang rejects them on arm64-apple-darwin. No wheels.

Binding brew's libspeexdsp.dylib directly costs ~40 lines, removes the rotten
wrapper from the chain, and unlocks speex_preprocess with
SPEEX_PREPROCESS_SET_ECHO_STATE -- residual echo suppression, measured on the
probe's synthetic path as the difference between ~17dB (linear MDF filter
alone) and ~34dB total echo suppression, with near-end (user) speech
preservation identical (lag-searched correlation 0.955 vs 0.956) because the
preprocessor's gain acts on echo-dominated bands, not on the user's voice.

Quantization: speex processes int16. Float32 frames are scaled by 32767 on
the way in and 1/32768 on the way out; the quantization floor (~-90dBFS) sits
far below any room's noise floor, so the round-trip is inaudible and
irrelevant to VAD/ASR.

Threading: a SpeexEchoState adapts on every process() call and is not
thread-safe. One instance must be driven by ONE thread at a time -- in
production that is the PortAudio input-callback thread (cost measured
~0.4ms per 80ms frame, two orders under the budget; see
tests/test_aec.py::test_per_frame_cost_within_callback_budget).
Construction on a different thread than process() is fine (plain C state,
nothing thread-local).

The far-end stream contract (why AudioIO feeds a FIFO ring, not "latest
frame wins"): the MDF filter models the room as a convolution of the
reference STREAM. Every sample the speaker actually played -- including
underrun zero-fill -- must reach process() exactly once, in order. Skipped
or duplicated reference frames look like a time-varying room and blow up
adaptation.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os

import numpy as np

SPEEX_ECHO_SET_SAMPLING_RATE = 24
SPEEX_PREPROCESS_SET_ECHO_STATE = 24

# Apple Silicon / Intel brew paths; find_library covers a system-installed lib.
_DYLIB_CANDIDATES = (
    "/opt/homebrew/opt/speexdsp/lib/libspeexdsp.dylib",
    "/usr/local/opt/speexdsp/lib/libspeexdsp.dylib",
)

_lib: ctypes.CDLL | None = None


def _load_speexdsp() -> ctypes.CDLL:
    """Load and configure libspeexdsp once per process."""
    global _lib
    if _lib is not None:
        return _lib
    path = ctypes.util.find_library("speexdsp")
    if path is None:
        path = next((p for p in _DYLIB_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        raise OSError(
            "libspeexdsp not found -- echo cancellation needs it: brew install speexdsp"
        )
    lib = ctypes.CDLL(path)
    lib.speex_echo_state_init.restype = ctypes.c_void_p
    lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
    lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.speex_echo_cancellation.argtypes = [ctypes.c_void_p] * 4
    lib.speex_preprocess_state_init.restype = ctypes.c_void_p
    lib.speex_preprocess_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.speex_preprocess_state_destroy.argtypes = [ctypes.c_void_p]
    lib.speex_preprocess_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.speex_preprocess_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib = lib
    return lib


class EchoCanceller:
    """Speex AEC over float32 frames: linear MDF filter + preprocessor stage.

    frame_samples is the outer frame AudioIO hands us (1920 = 80ms); speex
    adapts on chunk_samples sub-frames internally (480 = 20ms, the
    recommended speex analysis size -- the probe measured the best
    ERLE/cost balance there). filter_ms must cover the full delay spread
    between the reference tap (the output callback) and the echo's arrival
    in the mic: output device buffer (1-2 blocks = 80-160ms) + room flight
    + input buffer, plus the stream-start offset between the two callbacks.
    400ms covers it with margin; cost is linear in filter length and
    measured ~0.5ms per frame all-in.
    """

    def __init__(
        self,
        frame_samples: int = 1920,
        sample_rate: int = 24000,
        filter_ms: int = 400,
        chunk_samples: int = 480,
        preprocess: bool = True,
    ) -> None:
        if frame_samples % chunk_samples != 0:
            raise ValueError(
                f"frame_samples ({frame_samples}) must be a multiple of "
                f"chunk_samples ({chunk_samples})"
            )
        self._lib = _load_speexdsp()
        self.frame_samples = frame_samples
        self.chunk = chunk_samples
        filter_samples = int(filter_ms * sample_rate / 1000)
        self._echo_st = self._lib.speex_echo_state_init(chunk_samples, filter_samples)
        rate = ctypes.c_int(sample_rate)
        self._lib.speex_echo_ctl(
            self._echo_st, SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(rate)
        )
        self._pp_st = None
        if preprocess:
            self._pp_st = self._lib.speex_preprocess_state_init(chunk_samples, sample_rate)
            self._lib.speex_preprocess_ctl(
                self._pp_st, SPEEX_PREPROCESS_SET_ECHO_STATE, self._echo_st
            )
        # reused per-chunk int16 work buffers (the callback thread allocates
        # only the float32 result frame per call)
        self._mic16 = np.empty(chunk_samples, dtype=np.int16)
        self._ref16 = np.empty(chunk_samples, dtype=np.int16)
        self._out16 = np.empty(chunk_samples, dtype=np.int16)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Cancel ref's echo out of mic. Both float32, frame_samples long;
        returns a new float32 array. Single-thread use (see module docstring)."""
        out = np.empty(self.frame_samples, dtype=np.float32)
        c = self.chunk
        for i in range(0, self.frame_samples, c):
            np.clip(mic[i:i + c] * 32767.0, -32768, 32767, out=self._mic16, casting="unsafe")
            np.clip(ref[i:i + c] * 32767.0, -32768, 32767, out=self._ref16, casting="unsafe")
            self._lib.speex_echo_cancellation(
                self._echo_st,
                self._mic16.ctypes.data_as(ctypes.c_void_p),
                self._ref16.ctypes.data_as(ctypes.c_void_p),
                self._out16.ctypes.data_as(ctypes.c_void_p),
            )
            if self._pp_st is not None:
                self._lib.speex_preprocess_run(
                    self._pp_st, self._out16.ctypes.data_as(ctypes.c_void_p)
                )
            out[i:i + c] = self._out16
        out *= 1.0 / 32768.0
        return out

    def close(self) -> None:
        """Free the C states. Idempotent; the instance is dead afterwards."""
        if self._pp_st is not None:
            self._lib.speex_preprocess_state_destroy(self._pp_st)
            self._pp_st = None
        if self._echo_st is not None:
            self._lib.speex_echo_state_destroy(self._echo_st)
            self._echo_st = None

    def __del__(self) -> None:  # last-resort leak guard; close() is the real path
        try:
            self.close()
        except Exception:
            pass
