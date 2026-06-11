import asyncio

import pytest

from moneypenny.router import RouteDecision
from moneypenny.tools import ToolHost
from moneypenny.tools.homey_adapter import HomeyResult
from moneypenny.tools.timers import TimerService


class StubHomey:
    def __init__(self):
        self.kwargs = None

    def execute(self, **kwargs):
        self.kwargs = kwargs
        return HomeyResult(ok=True, summary="DONE DESK LAMP")


class Cfg:
    weather_lat, weather_lon = 25.2048, 55.2708


@pytest.fixture
def host():
    homey = StubHomey()
    # Python 3.12: get_event_loop() raises in sync tests; a fresh loop is enough
    # because these tests never need timers to actually fire.
    loop = asyncio.new_event_loop()
    timers = TimerService(on_fire=lambda l: None, loop=loop)
    yield ToolHost(Cfg(), homey, timers), homey
    loop.close()


def test_tier0_returns_none(host):
    h, _ = host
    assert h.execute(RouteDecision(0, None, {}, 0.9)) is None


def test_homey_dispatch_passes_structured_args(host):
    h, homey = host
    args = {"action": "turn_off", "device": "desk lamp", "zone": None,
            "capability": None, "value": None}
    out = h.execute(RouteDecision(1, "homey", args, 0.9))
    assert out.startswith("BRIEFING: DONE DESK LAMP")
    assert homey.kwargs == args


def test_homey_missing_action_escalates_not_crashes(host):
    h, homey = host
    out = h.execute(RouteDecision(1, "homey", {"device": "lamp"}, 0.9))
    assert "UNCLEAR" in out
    assert homey.kwargs is None  # nothing executed


async def test_timer_dispatch(host):
    h, _ = host
    out = h.execute(RouteDecision(1, "timer", {"duration": "5 minutes", "label": "tea"}, 0.9))
    assert "TIMER SET 5 MINUTES" in out


def test_unparseable_timer_briefs_for_clarification(host):
    h, _ = host
    out = h.execute(RouteDecision(1, "timer", {"duration": "a while"}, 0.9))
    assert "UNCLEAR" in out


def test_null_duration_does_not_crash(host):
    h, _ = host
    out = h.execute(RouteDecision(1, "timer", {"duration": None}, 0.9))
    assert "UNCLEAR" in out
