import asyncio
import threading

from moneypenny.tools.timers import TimerService, parse_duration


def test_parse_duration():
    assert parse_duration("5 minutes") == 300
    assert parse_duration("90 seconds") == 90
    assert parse_duration("1 hour") == 3600
    assert parse_duration("an hour and a half") is None  # unsupported -> escalate, don't guess
    assert parse_duration(None) is None  # router may emit null


async def test_timer_fires_callback():
    fired = []
    svc = TimerService(on_fire=lambda label: fired.append(label),
                       loop=asyncio.get_running_loop())
    svc.set_timer(0.05, "tea")
    await asyncio.sleep(0.15)
    assert fired == ["tea"]


async def test_set_timer_from_foreign_thread():
    """The production path: set_timer is called from a worker thread."""
    fired = []
    svc = TimerService(on_fire=lambda label: fired.append(label),
                       loop=asyncio.get_running_loop())
    t = threading.Thread(target=svc.set_timer, args=(0.05, "from-thread"))
    t.start()
    t.join()
    await asyncio.sleep(0.2)
    assert fired == ["from-thread"]


async def test_active_timers_listed_and_removed():
    svc = TimerService(on_fire=lambda label: None,
                       loop=asyncio.get_running_loop())
    svc.set_timer(10, "long")
    await asyncio.sleep(0.05)  # let the threadsafe schedule land
    assert [t.label for t in svc.active()] == ["long"]
    svc.cancel_all()
    await asyncio.sleep(0.05)
    assert svc.active() == []
