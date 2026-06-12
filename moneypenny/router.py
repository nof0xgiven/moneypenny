"""Latency-tier router: one structured-output call on a small local model.

Dumb, fast, escalation-biased (spec D4): any parse failure, low confidence,
or unknown tool resolves to Tier 2. Never guesses Tier 1.

Homey args are STRUCTURED (action/device/zone/capability/value) because the
resolver API requires an explicit action and routes dim/typed values through
its generic capability path - free text alone cannot drive it.
"""
from __future__ import annotations

import importlib
import json
import math
import re
import threading

import mlx.core as mx
from mlx_lm import generate, load

from moneypenny.route_decision import RouteDecision

# `import mlx_lm.generate as ...` would bind the generate FUNCTION (mlx_lm's
# __init__ re-exports it, shadowing the submodule attribute); we need the
# module itself to rebind its generation_stream global.
_mlx_lm_generate = importlib.import_module("mlx_lm.generate")

TOOLS = ("weather", "homey", "timer")
CONFIDENCE_FLOOR = 0.7

_SYSTEM = """You classify a voice assistant utterance into a latency tier. Reply with ONLY a JSON object, no other text:
{"tier": 0|1|2|3, "tool": "weather"|"homey"|"timer"|null, "args": {}, "confidence": 0.0-1.0}

tier 0: chat, banter, opinions, stories, and backchannels/acknowledgements ("yeah", "uh huh", "okay cool"). No facts needed. tool=null.
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

# Prompt edits here are EMPIRICAL: rerun tests/test_router.py -m "" (greedy
# decode = deterministic). Interactions are non-linear on this 0.6B model:
# backchannel acknowledgements belong in the tier 0 system line above, NOT in
# a fewshot pair - a ("yeah", tier 0) example alongside the garbled-weather
# pair below flips "tell me a story about a lighthouse" to tier 1 timer
# regardless of placement. Every pair also costs prompt tokens on every
# classification (latency budget: p95 < 300ms).
_FEWSHOT = [
    ("what's the weather today",
     '{"tier": 1, "tool": "weather", "args": {}, "confidence": 0.97}'),
    # ASR-garbled weather ask: disfluencies must not mask a recognizable
    # weather question (without this pair it classifies as tier 0 chat).
    ("uh I just wondered what the weather is",
     '{"tier": 1, "tool": "weather", "args": {}, "confidence": 0.9}'),
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

# MLX streams are thread-bound (per-thread stream registries), and
# mlx_lm.generate creates its module-level generation_stream at IMPORT time —
# i.e. on whatever thread first imports mlx_lm (the main thread, since this
# module imports it at top). generate() wraps all its work in
# `with mx.stream(generation_stream)`, so calling classify() from any other
# thread (the app's route worker) dies with
# "There is no Stream(gpu, N) in current thread." unless the global is
# rebound to a stream created on the calling thread. There is no public knob
# for this in mlx_lm; rebinding the module global is the supported-by-shape
# escape hatch. Serialized callers only: concurrent classify() from multiple
# threads was never supported (the app uses a single route worker).
_stream_owner: int | None = None


def _ensure_generation_stream_on_this_thread() -> None:
    global _stream_owner
    me = threading.get_ident()
    if _stream_owner != me:
        _mlx_lm_generate.generation_stream = mx.new_stream(mx.default_device())
        _stream_owner = me


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
        _ensure_generation_stream_on_this_thread()
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
