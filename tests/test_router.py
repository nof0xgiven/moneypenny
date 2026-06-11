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
