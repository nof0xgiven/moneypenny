"""Session lifecycle behavior that's testable without audio devices or models.

The frame loop itself needs real hardware; these tests exercise the crash
path by installing a real (failing) task as the loop task — the exact
mechanism start() uses — so nothing is mocked. The concurrency tests below
swap AudioIO/engine for tiny stand-ins at the sanctioned device/model
boundary (same policy as FakeSession in test_web.py): everything else —
start/stop bodies, the lifecycle lock, the frame-loop task, the bus — is
the real code.
"""
import asyncio
import queue
import time

import pytest

from moneypenny.app import Session
from moneypenny.config import Config
from moneypenny.events import EventBus


async def _crashed_session(bus: EventBus) -> Session:
    """Session whose 'frame loop' task died with a real exception."""
    s = Session(Config.from_env(), bus)

    async def boom() -> None:
        raise RuntimeError("kaboom")

    task = asyncio.get_running_loop().create_task(boom())
    task.add_done_callback(s._on_frame_loop_done)
    s._loop_task = task
    await asyncio.sleep(0.05)  # let the task die and callbacks/emits land
    return s


async def test_frame_loop_crash_emits_session_error():
    bus = EventBus(asyncio.get_running_loop())
    q = bus.subscribe()
    s = await _crashed_session(bus)
    assert not s.running
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    errors = [e for e in events if e["type"] == "session" and e["state"] == "error"]
    assert len(errors) == 1
    assert "kaboom" in errors[0]["reason"]
    # the dashboard's replay must not be stuck on a stale state
    assert bus.last("session")["state"] == "error"


async def test_start_tolerates_previously_crashed_task():
    bus = EventBus(asyncio.get_running_loop())
    s = await _crashed_session(bus)
    # A crashed task must read as not-running: start() proceeds past the
    # already-started guard (and then fails on the not-loaded guard, since
    # this test never loads models — proving the crash guard was cleared).
    with pytest.raises(RuntimeError, match="not loaded"):
        await s.start()
    assert s._loop_task is None  # crash debris cleaned up via the stop path


# --- lifecycle serialization (start/stop must not interleave at awaits) ---


class _FakeAudio:
    """Device-boundary stand-in: start() opens real audio; tests must not."""

    def __init__(self) -> None:
        self.mic_frames = queue.Queue()
        self.speaker_frames = queue.Queue()
        self.underruns = 0
        self.closes = 0

    def __enter__(self) -> "_FakeAudio":
        return self

    def __exit__(self, *exc) -> None:
        self.closes += 1


class _SlowResetEngine:
    """Model-boundary stand-in. reset_session sleeps on the engine pool so
    stop() parks at its run_in_executor await — the exact window where an
    unserialized start()/stop() interleaves (in prod the reset takes seconds)."""

    def __init__(self) -> None:
        self.resets = 0

    def reset_session(self) -> None:
        time.sleep(0.05)
        self.resets += 1


def _startable_session(bus: EventBus, monkeypatch) -> Session:
    monkeypatch.setattr("moneypenny.app.AudioIO", _FakeAudio)
    s = Session(Config.from_env(), bus)
    s.engine = _SlowResetEngine()  # passes the not-loaded guard; real reset path
    return s


async def _drain_pending() -> None:
    """Let call_soon_threadsafe emits scheduled so far land."""
    await asyncio.sleep(0.05)


async def test_concurrent_starts_exactly_one_succeeds(monkeypatch):
    bus = EventBus(asyncio.get_running_loop())
    s = _startable_session(bus, monkeypatch)
    results = await asyncio.gather(*(s.start() for _ in range(5)), return_exceptions=True)
    oks = [r for r in results if r is None]
    errs = [r for r in results if isinstance(r, RuntimeError)]
    assert len(oks) == 1
    assert len(errs) == 4
    assert all("already started" in str(e) for e in errs)
    assert s.running
    await s.stop()


async def test_start_during_inflight_stop_waits_for_stop(monkeypatch):
    """start() arriving while stop() is parked mid-teardown must wait for the
    full teardown — not interleave, close the new conversation's audio, and
    leave the bus's last session state lying as 'stopped' while live."""
    bus = EventBus(asyncio.get_running_loop())
    s = _startable_session(bus, monkeypatch)
    await s.start()
    stop_task = asyncio.get_running_loop().create_task(s.stop())
    await asyncio.sleep(0.01)  # stop() is now parked in reset_session (~50ms)
    await s.start()
    await stop_task
    await _drain_pending()
    assert s.running
    assert s._audio is not None  # the restarted conversation kept its audio
    assert s._audio.closes == 0
    assert bus.last("session")["state"] == "live"  # stopped emitted BEFORE live
    await s.stop()


async def test_concurrent_stops_are_idempotent(monkeypatch):
    bus = EventBus(asyncio.get_running_loop())
    s = _startable_session(bus, monkeypatch)
    await s.start()
    audio = s._audio
    q = bus.subscribe()
    await asyncio.gather(s.stop(), s.stop(), s.stop())
    await _drain_pending()
    assert not s.running
    assert audio.closes == 1
    assert s.engine.resets == 1  # later stops early-return, don't re-reset
    stopped = []
    while not q.empty():
        ev = q.get_nowait()
        if ev["type"] == "session" and ev["state"] == "stopped":
            stopped.append(ev)
    assert len(stopped) == 1
