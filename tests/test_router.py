"""Router classification on the real Qwen3 model. Slow, real, deliberate.

Escalation-bias contract: Tier 1 assertions are exact; ambiguous cases assert
NOT-tier-1 (they may be 0 or 2 - escalating is always acceptable)."""
import pytest

from moneypenny.router import Router, RouteDecision, parse_decision

TIER1_CASES = [
    ("what's the weather like today", "weather"),
    ("turn off the office lights", "homey"),
    ("set a timer for five minutes", "timer"),
    ("dim the bedroom lights to half", "homey"),
]

TIER0_OR_2_CASES = [
    "tell me a story about a lighthouse",        # chat
    "what do you think about that",              # chat
    "what's on my calendar this week",           # no calendar tool in phase 1 -> 2
    "research flight prices to tokyo and email me",  # tier 3 territory; 2 or 3 both fine
]


@pytest.fixture(scope="module")
def router():
    return Router()


@pytest.mark.slow
@pytest.mark.parametrize("utterance,tool", TIER1_CASES)
def test_tier1_reflex(router, utterance, tool):
    d = router.classify(utterance)
    assert d.tier == 1
    assert d.tool == tool


@pytest.mark.slow
def test_homey_args_are_structured(router):
    d = router.classify("dim the bedroom lights to half")
    assert d.tool == "homey"
    args = d.args
    # dim has no resolver action mapping; it must go through the capability path
    assert args.get("capability") == "dim"
    assert isinstance(args.get("value"), (int, float))
    assert "bedroom" in (args.get("device") or args.get("zone") or "").lower()


@pytest.mark.slow
@pytest.mark.parametrize("utterance", TIER0_OR_2_CASES)
def test_never_guesses_tier1(router, utterance):
    d = router.classify(utterance)
    assert d.tier != 1


@pytest.mark.slow
def test_garbage_input_escalates(router):
    d = router.classify("uh the thing with the the")
    assert d.tier in (0, 2)


def test_decision_parse_fallback_is_tier2():
    d = parse_decision("not json at all")
    assert d.tier == 2 and d.confidence == 0.0


def test_parse_low_confidence_tier1_escalates():
    d = parse_decision('{"tier": 1, "tool": "weather", "args": {}, "confidence": 0.69}')
    assert d.tier == 2 and d.tool is None


def test_parse_confidence_floor_boundary_keeps_tier1():
    d = parse_decision('{"tier": 1, "tool": "weather", "args": {}, "confidence": 0.7}')
    assert d.tier == 1 and d.tool == "weather"


def test_parse_unknown_tool_tier1_escalates():
    d = parse_decision('{"tier": 1, "tool": "calendar", "args": {}, "confidence": 0.95}')
    assert d.tier == 2 and d.tool is None


def test_parse_tier2_with_tool_set_nulls_tool():
    d = parse_decision('{"tier": 2, "tool": "weather", "args": {}, "confidence": 0.9}')
    assert d.tier == 2 and d.tool is None


def test_parse_invalid_tier_falls_back():
    d = parse_decision('{"tier": 5, "tool": null, "args": {}, "confidence": 0.9}')
    assert d.tier == 2 and d.confidence == 0.0


def test_parse_non_dict_args_falls_back():
    d = parse_decision('{"tier": 1, "tool": "weather", "args": [1, 2], "confidence": 0.9}')
    assert d.tier == 2 and d.confidence == 0.0


def test_parse_multiple_json_objects_never_raises():
    # Greedy regex spans both objects -> invalid JSON -> tier 2 fallback, no raise.
    raw = ('{"tier": 0, "tool": null, "args": {}, "confidence": 0.9} '
           '{"tier": 1, "tool": "weather", "args": {}, "confidence": 0.9}')
    d = parse_decision(raw)
    assert d.tier == 2


def test_parse_nan_confidence_escalates():
    # Python's json module accepts the NaN literal; NaN < floor is False, so an
    # unguarded comparison would keep tier 1 with NaN confidence.
    d = parse_decision('{"tier": 1, "tool": "weather", "args": {}, "confidence": NaN}')
    assert d.tier == 2 and d.confidence == 0.0
