"""ToolHost: executes a Tier 1 RouteDecision; returns briefing-ready text or None.

Action > narration: executions happen here regardless of what the voice
pipeline later does with the summary. Malformed args never raise - but they
no longer always brief either. A tier-1 route whose arguments fail validation
is more likely a router misroute on garbled ASR than a real request (the
router is an LLM; trust nothing), so the "ASK USER TO REPEAT" clarifications
are injected only when the transcript itself evidences the tool intent (a
trigger word, _TOOL_TRIGGERS below). No evidence -> drop silently (warning
log, return None): never ask the user to repeat a request they never made.
The adapter-absence briefings (NOT SET UP / UNAVAILABLE) are not
arg-validation failures and stay unconditional.

Weather is the exception to "never raise": network failures from
current_weather propagate to the caller; the app layer (moneypenny.app)
catches and converts them to failure briefings.
"""
from __future__ import annotations

import logging
import re

from moneypenny.briefing import compose
from moneypenny.route_decision import RouteDecision
from moneypenny.tools.timers import TimerService, parse_duration
from moneypenny.tools.weather import current_weather, format_weather

log = logging.getLogger("moneypenny.tools")


# Briefing per reason the homey adapter is absent. ToolHost stays dumb: the
# app decides the status; this is just a lookup.
_NO_HOMEY_BRIEFINGS = {
    "unconfigured": "HOME CONTROL NOT SET UP CANNOT DO THAT",
    "unavailable": "HOME CONTROL UNAVAILABLE RIGHT NOW CANNOT DO THAT",
}

# Words whose presence in the transcript counts as evidence the user really
# addressed the tool. Matched whole-word against the lowercased transcript.
_TOOL_TRIGGERS = {
    "timer": frozenset({"timer", "timers", "remind", "reminder", "reminders",
                        "countdown", "alarm", "alarms"}),
    "homey": frozenset({"light", "lights", "lamp", "lamps", "dim", "turn",
                        "switch", "plug", "heat", "heating", "thermostat"}),
}

# Filler inside a router-extracted device/zone string ("thing in the corner");
# finding these in the transcript evidences nothing.
_STOPWORDS = frozenset({"the", "a", "an", "in", "on", "of", "to", "and", "my"})


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _has_intent_evidence(tool: str, args: dict, transcript: str | None) -> bool:
    """Did the user plausibly ask for `tool` at all? True iff the transcript
    contains a trigger word for the tool - or, for homey, a word of the
    device/zone string the router extracted (a name appearing verbatim in the
    transcript means the user did address the home, trigger word or not)."""
    if not transcript:
        return False
    triggers = set(_TOOL_TRIGGERS[tool])
    if tool == "homey":
        for key in ("device", "zone"):
            v = args.get(key)
            if isinstance(v, str):
                triggers |= _words(v) - _STOPWORDS
    return bool(_words(transcript) & triggers)


class ToolHost:
    def __init__(self, cfg, homey_adapter, timer_service: TimerService,
                 homey_status: str = "unconfigured") -> None:
        self._cfg = cfg
        self._homey = homey_adapter
        self._timers = timer_service
        self._no_homey_briefing = _NO_HOMEY_BRIEFINGS[homey_status]

    def execute(self, decision: RouteDecision, transcript: str | None = None) -> str | None:
        """Returns a composed briefing string, or None when there is nothing to say.

        `transcript` is the user utterance that produced `decision`; it gates
        the arg-failure clarification briefings (see module docstring).
        transcript=None deliberately counts as NO evidence - drop, don't ask -
        because an unwitnessed "please repeat" is exactly the non-sequitur
        this policy exists to prevent. The production caller
        (moneypenny.app classify_and_execute) always has the transcript.
        """
        if decision.tier != 1:
            return None
        if decision.tool == "weather":
            w = current_weather(self._cfg.weather_lat, self._cfg.weather_lon)
            return compose("briefing", format_weather(w))
        if decision.tool == "homey":
            if self._homey is None:
                # No adapter: there is no action possible, so the useful
                # outcome IS the spoken explanation (set up vs. unreachable).
                return compose("briefing", self._no_homey_briefing)
            args = decision.args
            action = args.get("action")
            device = args.get("device")
            zone = args.get("zone")
            capability = args.get("capability")
            value = args.get("value")
            well_typed = (
                isinstance(action, str)
                and action
                and (device is None or isinstance(device, str))
                and (zone is None or isinstance(zone, str))
                and (device or zone)
                and (capability is None or isinstance(capability, str))
                and (value is None or isinstance(value, (bool, int, float, str)))
            )
            if not well_typed:
                if not _has_intent_evidence("homey", args, transcript):
                    log.warning("spurious homey route dropped (no tool evidence "
                                "in transcript): %r", transcript)
                    return None
                return compose("briefing", "HOMEY COMMAND UNCLEAR ASK USER TO REPEAT")
            result = self._homey.execute(
                action=action,
                device=device,
                zone=zone,
                capability=capability,
                value=value,
            )
            return compose("briefing", result.summary)
        if decision.tool == "timer":
            duration = decision.args.get("duration")
            seconds = parse_duration(duration)
            if seconds is None:
                if not _has_intent_evidence("timer", decision.args, transcript):
                    log.warning("spurious timer route dropped (no tool evidence "
                                "in transcript): %r", transcript)
                    return None
                return compose("briefing", "TIMER DURATION UNCLEAR ASK USER TO REPEAT")
            label = str(decision.args.get("label") or "timer")
            self._timers.set_timer(seconds, label)
            return compose("briefing", f"TIMER SET {duration.upper()} LABEL {label.upper()}")
        return None
