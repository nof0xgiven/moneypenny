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


def test_homey_decision_with_no_adapter_briefs_not_set_up():
    loop = asyncio.new_event_loop()
    try:
        timers = TimerService(on_fire=lambda l: None, loop=loop)
        h = ToolHost(Cfg(), None, timers)
        args = {"action": "turn_off", "device": "desk lamp"}
        out = h.execute(RouteDecision(1, "homey", args, 0.9))
        assert out.startswith("BRIEFING:")
        assert "NOT SET UP" in out
    finally:
        loop.close()


def test_homey_unavailable_status_briefs_unavailable_not_unconfigured():
    loop = asyncio.new_event_loop()
    try:
        timers = TimerService(on_fire=lambda l: None, loop=loop)
        h = ToolHost(Cfg(), None, timers, homey_status="unavailable")
        args = {"action": "turn_off", "device": "desk lamp"}
        out = h.execute(RouteDecision(1, "homey", args, 0.9))
        assert "UNAVAILABLE RIGHT NOW" in out
        assert "NOT SET UP" not in out
    finally:
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


def test_homey_non_string_device_escalates_not_crashes(host):
    h, homey = host
    out = h.execute(RouteDecision(1, "homey", {"action": "turn_on", "device": 42}, 0.9))
    assert "UNCLEAR" in out
    assert homey.kwargs is None


def test_homey_non_string_action_escalates_not_crashes(host):
    h, homey = host
    out = h.execute(RouteDecision(1, "homey", {"action": ["turn_on"], "zone": "office"}, 0.9))
    assert "UNCLEAR" in out
    assert homey.kwargs is None


def test_homey_non_string_capability_escalates_not_crashes(host):
    h, homey = host
    args = {"action": "set", "zone": "office", "capability": ["dim"], "value": 0.5}
    out = h.execute(RouteDecision(1, "homey", args, 0.9))
    assert "UNCLEAR" in out
    assert homey.kwargs is None


def test_homey_non_scalar_value_escalates_not_crashes(host):
    h, homey = host
    args = {"action": "set", "zone": "office", "capability": "dim", "value": {"level": 0.5}}
    out = h.execute(RouteDecision(1, "homey", args, 0.9))
    assert "UNCLEAR" in out
    assert homey.kwargs is None


def test_timer_weird_label_does_not_crash(host):
    h, _ = host
    out = h.execute(RouteDecision(1, "timer", {"duration": "2 minutes", "label": {"x": 1}}, 0.9))
    assert "TIMER SET 2 MINUTES" in out


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
