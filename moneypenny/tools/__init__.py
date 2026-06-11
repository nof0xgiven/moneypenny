"""ToolHost: executes a Tier 1 RouteDecision; returns briefing-ready text or None.

Action > narration: executions happen here regardless of what the voice
pipeline later does with the summary. Malformed args never raise - they
produce clarification briefings (the router is an LLM; trust nothing).

Weather is the exception to "never raise": network failures from
current_weather propagate to the caller; the app layer (moneypenny.app)
catches and converts them to failure briefings.
"""
from __future__ import annotations

from moneypenny.briefing import compose
from moneypenny.router import RouteDecision
from moneypenny.tools.timers import TimerService, parse_duration
from moneypenny.tools.weather import current_weather, format_weather


class ToolHost:
    def __init__(self, cfg, homey_adapter, timer_service: TimerService) -> None:
        self._cfg = cfg
        self._homey = homey_adapter
        self._timers = timer_service

    def execute(self, decision: RouteDecision) -> str | None:
        """Returns a composed briefing string, or None when there is nothing to say."""
        if decision.tier != 1:
            return None
        if decision.tool == "weather":
            w = current_weather(self._cfg.weather_lat, self._cfg.weather_lon)
            return compose("briefing", format_weather(w))
        if decision.tool == "homey":
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
                return compose("briefing", "TIMER DURATION UNCLEAR ASK USER TO REPEAT")
            label = str(decision.args.get("label") or "timer")
            self._timers.set_timer(seconds, label)
            return compose("briefing", f"TIMER SET {duration.upper()} LABEL {label.upper()}")
        return None
