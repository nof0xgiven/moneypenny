import asyncio
import logging

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
    out = h.execute(RouteDecision(1, "homey", {"device": "lamp"}, 0.9),
                    transcript="do something with the lamp")
    assert "UNCLEAR" in out
    assert homey.kwargs is None  # nothing executed


def test_homey_non_string_device_escalates_not_crashes(host):
    h, homey = host
    out = h.execute(RouteDecision(1, "homey", {"action": "turn_on", "device": 42}, 0.9),
                    transcript="turn on the whatsit")
    assert "UNCLEAR" in out
    assert homey.kwargs is None


def test_homey_non_string_action_escalates_not_crashes(host):
    h, homey = host
    out = h.execute(RouteDecision(1, "homey", {"action": ["turn_on"], "zone": "office"}, 0.9),
                    transcript="switch everything in the office")
    assert "UNCLEAR" in out
    assert homey.kwargs is None


def test_homey_non_string_capability_escalates_not_crashes(host):
    h, homey = host
    args = {"action": "set", "zone": "office", "capability": ["dim"], "value": 0.5}
    out = h.execute(RouteDecision(1, "homey", args, 0.9),
                    transcript="dim the office a bit")
    assert "UNCLEAR" in out
    assert homey.kwargs is None


def test_homey_non_scalar_value_escalates_not_crashes(host):
    h, homey = host
    args = {"action": "set", "zone": "office", "capability": "dim", "value": {"level": 0.5}}
    out = h.execute(RouteDecision(1, "homey", args, 0.9),
                    transcript="set the office lights to about half")
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
    out = h.execute(RouteDecision(1, "timer", {"duration": "a while"}, 0.9),
                    transcript="set a timer for a while")
    assert "UNCLEAR" in out


def test_null_duration_does_not_crash(host):
    h, _ = host
    out = h.execute(RouteDecision(1, "timer", {"duration": None}, 0.9),
                    transcript="set a timer")
    assert "UNCLEAR" in out


# --- clarification restraint: arg-failure briefings need transcript evidence
#     (a tier-1 route whose args fail validation is more likely a router
#     misroute than a real request; only ask back when the user plausibly
#     asked at all)


def test_spurious_timer_route_dropped_silently(host, caplog):
    """The production incident verbatim: ASR garbled chat into a timer route."""
    h, homey = host
    args = {"duration": "function", "label": "function"}
    with caplog.at_level(logging.WARNING):
        out = h.execute(RouteDecision(1, "timer", args, 0.95),
                        transcript="I don't mind the function")
    assert out is None  # no "say it again?" about a timer nobody asked for
    assert homey.kwargs is None
    assert any("spurious timer route dropped" in r.message for r in caplog.records)


def test_timer_evidence_keeps_clarification_for_real_garbled_request(host):
    h, _ = host
    out = h.execute(RouteDecision(1, "timer", {"duration": "elevenses"}, 0.9),
                    transcript="set a timer for elevenses")
    assert "UNCLEAR" in out  # user said "timer"; asking back is good UX


def test_homey_missing_action_with_trigger_word_keeps_clarification(host):
    h, homey = host
    out = h.execute(RouteDecision(1, "homey", {"device": "thing"}, 0.9),
                    transcript="turn on the thing in the corner")
    assert "UNCLEAR" in out
    assert homey.kwargs is None


def test_homey_missing_action_without_evidence_dropped(host, caplog):
    h, homey = host
    with caplog.at_level(logging.WARNING):
        out = h.execute(RouteDecision(1, "homey", {"device": "lamp"}, 0.9),
                        transcript="that's what she said")
    assert out is None
    assert homey.kwargs is None
    assert any("spurious homey route dropped" in r.message for r in caplog.records)


def test_homey_device_word_alone_is_not_evidence(host, caplog):
    # Flipped by review: the router COPIES transcript words into device/zone
    # args, so a copied arg appearing verbatim proves nothing (see the
    # "function" test below for the failure that killed the rule). Evidence
    # is trigger words only. Accepted cost: a real terse request like this
    # one degrades to silence and the user's natural retry - a phantom
    # "please repeat" is the worse failure.
    h, homey = host
    with caplog.at_level(logging.WARNING):
        out = h.execute(RouteDecision(1, "homey", {"device": "kettle"}, 0.9),
                        transcript="kettle off please")
    assert out is None
    assert homey.kwargs is None
    assert any("spurious homey route dropped" in r.message for r in caplog.records)


def test_homey_copied_arg_words_are_not_evidence(host, caplog):
    # The production failure shape, homey edition: the router invents
    # device="function" out of a chat sentence, so the arg's words trivially
    # appear verbatim in the transcript. That must not license a clarification.
    h, homey = host
    with caplog.at_level(logging.WARNING):
        out = h.execute(RouteDecision(1, "homey", {"device": "function"}, 0.95),
                        transcript="I don't mind the function")
    assert out is None
    assert homey.kwargs is None
    assert any("spurious homey route dropped" in r.message for r in caplog.records)


def test_no_transcript_drops_arg_failure_clarifications(host):
    # transcript=None means no evidence is checkable: the documented default
    # is to drop, never to ask the user to repeat something unwitnessed.
    h, _ = host
    out = h.execute(RouteDecision(1, "timer", {"duration": "a while"}, 0.9))
    assert out is None


def test_homey_not_set_up_briefing_is_unconditional():
    # NOT SET UP / UNAVAILABLE are not arg-validation failures: the args were
    # fine, the adapter is absent. They brief regardless of transcript.
    loop = asyncio.new_event_loop()
    try:
        timers = TimerService(on_fire=lambda l: None, loop=loop)
        h = ToolHost(Cfg(), None, timers)
        args = {"action": "turn_off", "device": "desk lamp"}
        out = h.execute(RouteDecision(1, "homey", args, 0.9),
                        transcript="that's what she said")
        assert "NOT SET UP" in out
    finally:
        loop.close()
