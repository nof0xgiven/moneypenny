"""Session lifecycle behavior that's testable without audio devices or models.

The frame loop itself needs real hardware; these tests exercise the crash
path by installing a real (failing) task as the loop task — the exact
mechanism start() uses — so nothing is mocked.
"""
import asyncio

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
