"""Latency-tier router: one structured-output call on a small local model.

Dumb, fast, escalation-biased (spec D4): any parse failure, low confidence,
or unknown tool resolves to Tier 2. Never guesses Tier 1.

Homey args are STRUCTURED (action/device/zone/capability/value) because the
resolver API requires an explicit action and routes dim/typed values through
its generic capability path - free text alone cannot drive it.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from mlx_lm import generate, load

TOOLS = ("weather", "homey", "timer")
CONFIDENCE_FLOOR = 0.7

_SYSTEM = """You classify a voice assistant utterance into a latency tier. Reply with ONLY a JSON object, no other text:
{"tier": 0|1|2|3, "tool": "weather"|"homey"|"timer"|null, "args": {}, "confidence": 0.0-1.0}

tier 0: chat, banter, opinions, stories. No facts needed. tool=null.
tier 1: a single reflex action -> tool required:
  weather: current weather questions. args: {}
  homey: home device control. args: {"action": "turn_on"|"turn_off"|"toggle"|"set",
    "device": "<device or group name, or null>", "zone": "<room name or null>",
    "capability": "<exact capability id like dim, or null>", "value": <number|bool|string|null>}
    Use action "set" with capability+value for dimming/levels (e.g. dim to half ->
    capability "dim", value 0.5). Plain on/off needs only action+device/zone.
  timer: set a timer with an explicit duration. args: {"duration": "<e.g. 5 minutes>", "label": "<short label>"}
tier 2: anything needing reasoning, lookups, calendars, email, memory, or that you are unsure about. tool=null.
tier 3: long-running tasks the user wants done in the background ("research X and email me"). tool=null.

If in ANY doubt, use tier 2. Never use tier 1 unless the utterance is unambiguous."""

_FEWSHOT = [
    ("what's the weather today",
     '{"tier": 1, "tool": "weather", "args": {}, "confidence": 0.97}'),
    ("turn off the kitchen lights",
     '{"tier": 1, "tool": "homey", "args": {"action": "turn_off", "device": "kitchen lights", '
     '"zone": "kitchen", "capability": null, "value": null}, "confidence": 0.95}'),
    ("dim the living room to twenty percent",
     '{"tier": 1, "tool": "homey", "args": {"action": "set", "device": null, '
     '"zone": "living room", "capability": "dim", "value": 0.2}, "confidence": 0.9}'),
    ("set a timer for ten minutes",
     '{"tier": 1, "tool": "timer", "args": {"duration": "10 minutes", "label": "timer"}, "confidence": 0.95}'),
    ("tell me something funny",
     '{"tier": 0, "tool": null, "args": {}, "confidence": 0.9}'),
    ("do I have anything tomorrow morning",
     '{"tier": 2, "tool": null, "args": {}, "confidence": 0.85}'),
    ("look into good e-bikes and email me a shortlist",
     '{"tier": 3, "tool": null, "args": {}, "confidence": 0.9}'),
]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class RouteDecision:
    tier: int
    tool: str | None
    args: dict
    confidence: float


def parse_decision(raw: str) -> RouteDecision:
    """Parse model output; any defect -> Tier 2 (escalation bias)."""
    fallback = RouteDecision(tier=2, tool=None, args={}, confidence=0.0)
    m = _JSON_RE.search(raw)
    if not m:
        return fallback
    try:
        d = json.loads(m.group(0))
        tier = int(d["tier"])
        tool = d.get("tool")
        conf = float(d.get("confidence", 0.0))
        if not math.isfinite(conf):
            # NaN compares False against the floor and would sneak past it.
            conf = 0.0
        args = d.get("args") or {}
        if tier not in (0, 1, 2, 3) or not isinstance(args, dict):
            return fallback
        if tier == 1 and (tool not in TOOLS or conf < CONFIDENCE_FLOOR):
            return RouteDecision(tier=2, tool=None, args={}, confidence=conf)
        return RouteDecision(tier=tier, tool=tool if tier == 1 else None, args=args, confidence=conf)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


class Router:
    def __init__(self, model_id: str = "mlx-community/Qwen3-0.6B-4bit") -> None:
        self._model, self._tokenizer = load(model_id)

    def classify(self, transcript: str) -> RouteDecision:
        messages = [{"role": "system", "content": _SYSTEM}]
        for u, a in _FEWSHOT:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": transcript + " /no_think"})
        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
        raw = generate(self._model, self._tokenizer, prompt=prompt, max_tokens=120, verbose=False)
        return parse_decision(raw)
