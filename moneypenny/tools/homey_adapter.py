"""Homey adapter: structured router args -> resolver plan -> client execution.

Replaces the Gemini-coupled tool layer from the source repo.
Tier 1 policy: requires_confirmation plans are refused (not reflex material);
no-match returns ok=False with a briefing-ready summary. Never raises to callers.
"""
from __future__ import annotations

from dataclasses import dataclass

from moneypenny.homey.aliases import HomeyAliases
from moneypenny.homey.catalog import HomeyCatalog
from moneypenny.homey.client import HomeyClient, HomeyClientError
from moneypenny.homey.resolver import HomeyResolver


@dataclass(frozen=True)
class HomeyResult:
    ok: bool
    summary: str  # briefing-ready, telegraphic


class HomeyAdapter:
    def __init__(self, client, resolver: HomeyResolver) -> None:
        self._client = client
        self._resolver = resolver

    @classmethod
    def from_config(cls, cfg) -> "HomeyAdapter":
        client = HomeyClient(base_url=cfg.homey_base_url, api_key=cfg.homey_api_key)
        catalog = HomeyCatalog.from_payloads(client.get_devices(), client.get_zones())
        return cls(client, HomeyResolver(catalog, HomeyAliases.load()))

    def execute(
        self,
        *,
        action: str,
        device: str | None = None,
        zone: str | None = None,
        capability: str | None = None,
        value: bool | int | float | str | None = None,
    ) -> HomeyResult:
        command = " ".join(p for p in (action, device or "", zone or "") if p)
        kwargs: dict = {
            "command": command,
            "action": action,
            "device": device,
            "zone": zone,
            "capability": capability,
        }
        if isinstance(value, bool):
            kwargs["value_bool"] = value
        elif isinstance(value, (int, float)):
            kwargs["value_number"] = value
        elif isinstance(value, str) and value:
            kwargs["value_text"] = value

        plan = self._resolver.resolve(**kwargs)

        if plan.requires_confirmation:
            # High-impact / broad ops are not Tier-1 material; Phase 2 escalates these.
            return HomeyResult(False, f"NEEDS CONFIRMATION NOT DONE: {plan.message.upper()}")
        if plan.status_reads and not plan.operations:
            # Status queries resolve to reads, not writes; Phase 2 territory.
            return HomeyResult(False, "STATUS QUERY NOT SUPPORTED YET")
        if not plan.operations:
            reason = "; ".join(plan.failures) or plan.message or "no matching device"
            return HomeyResult(False, f"HOMEY FAILED {reason.upper()}")

        done: list[str] = []
        failed: list[str] = []
        for op in plan.operations:
            if op.already_done:
                done.append(op.device_name)
                continue
            try:
                self._client.set_capability_value(op.device_id, op.capability_id, op.value)
                done.append(op.device_name)
            except HomeyClientError as exc:
                failed.append(f"{op.device_name} ({exc})")

        if failed and not done:
            return HomeyResult(False, "HOMEY ERROR " + "; ".join(failed).upper())
        summary = "DONE " + " AND ".join(n.upper() for n in done)
        if failed:
            summary += " BUT FAILED " + "; ".join(failed).upper()
        return HomeyResult(True, summary)
