"""AsrGate: VAD-gated ASR feeding policy (pre-roll flush, hangover, batching).

Driven through a REAL EnergyVAD (defaults: soft=2, hard=8) so the tests pin the
actual event/in_speech semantics the gate sees in app.py, not a simulation.
Frames are constant-valued so emitted buffers can be checked sample-exactly.
"""
import numpy as np

from moneypenny.asr_gate import AsrGate
from moneypenny.vad import EnergyVAD

FRAME = 1920


def quiet(tag: float = 0.001) -> np.ndarray:
    """Below the 0.01 RMS threshold; tag makes frames distinguishable."""
    assert tag < 0.01
    return np.full(FRAME, tag, dtype=np.float32)


def loud(tag: float = 0.5) -> np.ndarray:
    assert tag >= 0.01
    return np.full(FRAME, tag, dtype=np.float32)


def drive(gate: AsrGate, vad: EnergyVAD, frames: list[np.ndarray]) -> list[np.ndarray | None]:
    out = []
    for f in frames:
        event = vad.feed(f)
        out.append(gate.feed(f, event, vad.in_speech))
    return out


def cat(*frames: np.ndarray) -> np.ndarray:
    return np.concatenate(frames)


def test_gated_off_feeds_nothing():
    gate, vad = AsrGate(), EnergyVAD()
    outs = drive(gate, vad, [quiet(0.001 * (i + 1) / 10) for i in range(20)])
    assert all(o is None for o in outs)


def test_speech_start_flushes_preroll_plus_current_frame():
    gate, vad = AsrGate(), EnergyVAD()
    qs = [quiet(0.0001 * (i + 1)) for i in range(6)]
    first = loud()
    outs = drive(gate, vad, qs + [first])
    assert all(o is None for o in outs[:6])
    # last 4 silence frames (320ms pre-roll) then the first loud frame
    np.testing.assert_array_equal(outs[6], cat(qs[2], qs[3], qs[4], qs[5], first))


def test_speech_start_with_partially_filled_preroll():
    gate, vad = AsrGate(), EnergyVAD()
    q1, q2, first = quiet(0.001), quiet(0.002), loud()
    outs = drive(gate, vad, [q1, q2, first])
    np.testing.assert_array_equal(outs[2], cat(q1, q2, first))


def test_batching_pairs_frames_during_speech():
    gate, vad = AsrGate(), EnergyVAD()
    ls = [loud(0.1 * (i + 1)) for i in range(5)]
    outs = drive(gate, vad, ls)
    np.testing.assert_array_equal(outs[0], ls[0])  # speech_start flush (empty pre-roll)
    assert outs[1] is None                          # buffered
    np.testing.assert_array_equal(outs[2], cat(ls[1], ls[2]))  # pair: 3840 samples
    assert outs[3] is None
    np.testing.assert_array_equal(outs[4], cat(ls[3], ls[4]))


def test_hangover_feeds_exactly_8_frames_past_speech_end():
    gate, vad = AsrGate(), EnergyVAD()  # hard boundary = 8 frames = hangover
    ls = [loud(0.1 * (i + 1)) for i in range(4)]
    ss = [quiet(0.0001 * (i + 1)) for i in range(12)]
    outs = drive(gate, vad, ls + ss)
    fed = np.concatenate([o for o in outs if o is not None])
    # every loud frame and EXACTLY the first 8 silent frames, in order, no gaps
    np.testing.assert_array_equal(fed, cat(*ls, *ss[:8]))
    assert all(o is None for o in outs[12:])  # silent frames 9..12 gated off
    assert not gate.active


def test_utterance_end_flushes_pending_frame():
    gate, vad = AsrGate(), EnergyVAD()
    # 5 loud: flush(L0), pend(L1), pair(L1,L2), pend(L3), pair(L3,L4)
    # 8 silent: pend(s1), pair(s1,s2), ... pend(s7) -> utterance_end on s8
    ls = [loud(0.1 * (i + 1)) for i in range(5)]
    ss = [quiet(0.0001 * (i + 1)) for i in range(8)]
    outs = drive(gate, vad, ls + ss)
    np.testing.assert_array_equal(outs[-1], cat(ss[6], ss[7]))  # pending + final frame
    fed = np.concatenate([o for o in outs if o is not None])
    np.testing.assert_array_equal(fed, cat(*ls, *ss))  # nothing lost, nothing doubled


def test_resumed_speech_after_maybe_end_keeps_feeding_without_reflush():
    gate, vad = AsrGate(), EnergyVAD()
    ls = [loud(0.1 * (i + 1)) for i in range(3)]
    pause = [quiet(0.0001), quiet(0.0002)]  # 2 silent frames -> maybe_end fires
    resume = [loud(0.4), loud(0.45)]
    outs = drive(gate, vad, ls + pause + resume)
    fed = np.concatenate([o for o in outs if o is not None])
    # pause and resumed frames all fed in order; no second pre-roll flush
    np.testing.assert_array_equal(fed[: 6 * FRAME], cat(*ls, *pause, resume[0]))


def test_preroll_refills_during_silence_for_next_utterance():
    gate, vad = AsrGate(), EnergyVAD()
    drive(gate, vad, [loud()] * 3 + [quiet(0.0001)] * 8)  # full utterance + hard end
    rs = [quiet(0.001 * (i + 1)) for i in range(5)]
    nxt = loud(0.7)
    outs = drive(gate, vad, rs + [nxt])
    assert all(o is None for o in outs[:5])
    # next utterance gets ONLY the fresh silence pre-roll, nothing stale
    np.testing.assert_array_equal(outs[5], cat(rs[1], rs[2], rs[3], rs[4], nxt))


def test_active_tracks_speech_and_hangover():
    gate, vad = AsrGate(), EnergyVAD()
    drive(gate, vad, [quiet()])
    assert not gate.active
    drive(gate, vad, [loud()])
    assert gate.active
    for i in range(7):  # silent frames 1..7: hangover, still active
        drive(gate, vad, [quiet()])
        assert gate.active
    drive(gate, vad, [quiet()])  # 8th silent frame: utterance_end
    assert not gate.active
