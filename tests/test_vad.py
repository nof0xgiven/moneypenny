import numpy as np

from moneypenny.vad import EnergyVAD

FRAME = 1920


def loud(n=FRAME):
    rng = np.random.default_rng(7)
    return (rng.standard_normal(n) * 0.3).astype(np.float32)


def quiet(n=FRAME):
    return np.zeros(n, dtype=np.float32)


def test_maybe_end_then_utterance_end():
    vad = EnergyVAD(soft_silence_frames=2, silence_frames=5)
    events = []
    for _ in range(10):
        events.append(vad.feed(loud()))
    assert vad.feed(quiet()) is None          # 1 silent frame: nothing yet
    assert vad.feed(quiet()) == "maybe_end"   # soft boundary at 2
    assert vad.feed(quiet()) is None
    assert vad.feed(quiet()) is None
    assert vad.feed(quiet()) == "utterance_end"  # hard boundary at 5


def test_resumed_speech_cancels_maybe_end():
    vad = EnergyVAD(soft_silence_frames=2, silence_frames=5)
    for _ in range(10):
        vad.feed(loud())
    vad.feed(quiet())
    assert vad.feed(quiet()) == "maybe_end"
    assert vad.feed(loud()) is None  # speech resumed; no utterance_end follows yet
    for _ in range(4):
        vad.feed(quiet())
    assert vad.feed(quiet()) == "utterance_end"


def test_no_event_during_pure_silence():
    vad = EnergyVAD()
    assert all(vad.feed(quiet()) is None for _ in range(20))


def test_speech_start_event():
    vad = EnergyVAD()
    vad.feed(quiet())
    assert vad.feed(loud()) == "speech_start"
