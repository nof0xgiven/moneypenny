"""EventBus: thread-safe pub/sub plumbing for the web dashboard (no models)."""
import asyncio
import threading
import time

import pytest

from moneypenny.events import EventBus


async def _drain_pending() -> None:
    """Let call_soon_threadsafe callbacks scheduled so far run."""
    await asyncio.sleep(0.05)


async def test_emit_on_loop_delivers():
    bus = EventBus(asyncio.get_running_loop())
    q = bus.subscribe()
    t0 = time.time()
    bus.emit("status", fps=12.5)
    ev = await asyncio.wait_for(q.get(), timeout=1)
    assert ev["type"] == "status"
    assert ev["fps"] == 12.5
    assert t0 <= ev["ts"] <= time.time()


async def test_emit_from_foreign_thread_delivers():
    """The production path: route/asr/tts workers emit from their own threads."""
    bus = EventBus(asyncio.get_running_loop())
    q = bus.subscribe()
    t = threading.Thread(target=bus.emit, args=("route",), kwargs={"tier": 1})
    t.start()
    t.join()
    ev = await asyncio.wait_for(q.get(), timeout=1)
    assert ev["type"] == "route"
    assert ev["tier"] == 1


async def test_two_subscribers_both_receive():
    bus = EventBus(asyncio.get_running_loop())
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    bus.emit("vad", event="speech_start")
    ev1 = await asyncio.wait_for(q1.get(), timeout=1)
    ev2 = await asyncio.wait_for(q2.get(), timeout=1)
    assert ev1["event"] == ev2["event"] == "speech_start"


async def test_overflow_drops_oldest_keeps_newest():
    bus = EventBus(asyncio.get_running_loop())
    q = bus.subscribe(maxsize=4)
    for i in range(10):
        bus.emit("audio", seq=i)
    await _drain_pending()
    got = []
    while not q.empty():
        got.append(q.get_nowait()["seq"])
    # oldest dropped, newest kept, order preserved
    assert got == [6, 7, 8, 9]
    assert q.dropped == 6


async def test_subscribe_rejects_unbounded_queue():
    """maxsize=0 means UNBOUNDED for asyncio.Queue, defeating the drop-oldest
    overflow guard — must raise even under python -O (so not an assert)."""
    bus = EventBus(asyncio.get_running_loop())
    with pytest.raises(ValueError, match="bounded"):
        bus.subscribe(maxsize=0)
    with pytest.raises(ValueError, match="bounded"):
        bus.subscribe(maxsize=-1)
    assert bus._subscribers == []  # nothing registered on the failed calls


async def test_last_returns_latest_per_type():
    bus = EventBus(asyncio.get_running_loop())
    bus.emit("status", fps=1.0)
    bus.emit("session", state="loading")
    bus.emit("status", fps=2.0)
    await _drain_pending()
    assert bus.last("status")["fps"] == 2.0
    assert bus.last("session")["state"] == "loading"
    assert bus.last("nope") is None


async def test_unsubscribe_stops_delivery():
    bus = EventBus(asyncio.get_running_loop())
    q = bus.subscribe()
    bus.emit("audio", seq=0)
    await _drain_pending()
    bus.unsubscribe(q)
    bus.emit("audio", seq=1)
    await _drain_pending()
    assert q.get_nowait()["seq"] == 0
    assert q.empty()
