from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .aliases import HomeyAliases, normalize_name
from .catalog import CapabilityRecord, DeviceRecord, HomeyCatalog


@dataclass(frozen=True)
class HomeyOperation:
    device_id: str
    device_name: str
    zone_name: str
    capability_id: str
    value: Any
    value_rule: str = ""
    high_impact: bool = False
    already_done: bool = False


@dataclass(frozen=True)
class HomeyStatusRead:
    device_id: str
    device_name: str
    zone_name: str
    capability_id: str


@dataclass(frozen=True)
class HomeyPlan:
    message: str
    operations: tuple[HomeyOperation, ...]
    status_reads: tuple[HomeyStatusRead, ...]
    requires_confirmation: bool = False
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ActionMapping:
    actions: tuple[str, ...]
    capability_ids: tuple[str, ...]
    value_rule: str
    value: Any = None
    requires_confirmation: bool = False


@dataclass(frozen=True)
class _StatusMapping:
    actions: tuple[str, ...]
    capability_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedTargets:
    devices: tuple[DeviceRecord, ...]
    requires_confirmation: bool = False
    clarification: str | None = None
    failures: tuple[str, ...] = ()


_BOOLEAN_TRUE = {"1", "true", "yes", "on", "enable", "enabled", "auto"}
_BOOLEAN_FALSE = {"0", "false", "no", "off", "disable", "disabled"}
_BROAD_WORDS = {"all", "everything"}
_BROAD_PHRASES = ("all devices", "whole home", "entire home")
_VALUE_SEPARATORS_RE = re.compile(r"[\s_-]+")
_HIGH_IMPACT_CLASSES = {"camera", "thermostat", "windowcoverings"}
_THERMOSTAT_CAPABILITIES = {
    "target_temperature.cool",
    "nest_thermostat_mode",
    "nest_thermostat_eco",
}
_CAMERA_CONTROL_CAPABILITIES = {
    "ptz_up",
    "ptz_down",
    "ptz_left",
    "ptz_right",
    "zoom_in",
    "zoom_out",
    "spotlight",
    "ir_lights",
    "infrared",
    "day_night",
    "day_night_mode",
}


_WRITE_MAPPINGS = (
    _ActionMapping(("turn_on", "on", "power_on"), ("onoff",), "literal", True),
    _ActionMapping(("turn_off", "off", "power_off"), ("onoff",), "literal", False),
    _ActionMapping(("toggle",), ("onoff",), "toggle"),
    _ActionMapping(("set_fan_speed",), ("state_fanspeed",), "typed"),
    _ActionMapping(("set_auto_mode",), ("state_automode",), "typed"),
    _ActionMapping(("set_fan_direction",), ("state_fandirection",), "typed"),
    _ActionMapping(("set_night_mode",), ("state_nightmode",), "typed"),
    _ActionMapping(
        ("set_continuous_monitoring",),
        ("state_continuousmonitoring",),
        "typed",
    ),
    _ActionMapping(("set_humidifier_mode",), ("state_humidifiermode",), "typed"),
    _ActionMapping(("set_target_humidity",), ("target_humidity",), "typed"),
    _ActionMapping(("set_oscillation_mode",), ("state_oscillationmode",), "typed"),
    _ActionMapping(("mute",), ("volume_mute",), "button", True),
    _ActionMapping(("volume_up",), ("volume_up",), "button", True),
    _ActionMapping(("volume_down",), ("volume_down",), "button", True),
    _ActionMapping(("channel_up",), ("channel_up",), "button", True),
    _ActionMapping(("channel_down",), ("channel_down",), "button", True),
    _ActionMapping(
        ("set_temperature",),
        ("target_temperature.cool",),
        "typed",
        requires_confirmation=True,
    ),
    _ActionMapping(
        ("set_thermostat_mode",),
        ("nest_thermostat_mode",),
        "typed",
        requires_confirmation=True,
    ),
    _ActionMapping(
        ("set_eco",),
        ("nest_thermostat_eco",),
        "typed",
        requires_confirmation=True,
    ),
    _ActionMapping(
        ("open", "up"),
        ("windowcoverings_state",),
        "literal",
        "up",
        requires_confirmation=True,
    ),
    _ActionMapping(
        ("close", "down"),
        ("windowcoverings_state",),
        "literal",
        "down",
        requires_confirmation=True,
    ),
    _ActionMapping(
        ("tilt_up",),
        ("windowcoverings_tilt_up",),
        "button",
        True,
        requires_confirmation=True,
    ),
    _ActionMapping(
        ("tilt_down",),
        ("windowcoverings_tilt_down",),
        "button",
        True,
        requires_confirmation=True,
    ),
    _ActionMapping(("ptz_up",), ("ptz_up",), "button", True, True),
    _ActionMapping(("ptz_down",), ("ptz_down",), "button", True, True),
    _ActionMapping(("ptz_left",), ("ptz_left",), "button", True, True),
    _ActionMapping(("ptz_right",), ("ptz_right",), "button", True, True),
    _ActionMapping(("zoom_in",), ("zoom_in",), "button", True, True),
    _ActionMapping(("zoom_out",), ("zoom_out",), "button", True, True),
    _ActionMapping(("spotlight",), ("spotlight",), "typed", requires_confirmation=True),
    _ActionMapping(
        ("ir_lights",),
        ("ir_lights", "infrared"),
        "typed",
        requires_confirmation=True,
    ),
    _ActionMapping(
        ("day_night",),
        ("day_night", "day_night_mode"),
        "typed",
        requires_confirmation=True,
    ),
)

_STATUS_MAPPINGS = (
    _StatusMapping(
        ("status", "get_status", "get_power_status", "power_status"),
        ("onoff",),
    ),
    _StatusMapping(
        ("get_temperature", "temperature_status"),
        ("measure_temperature", "target_temperature.cool"),
    ),
    _StatusMapping(
        ("get_target_temperature", "target_temperature_status"),
        ("target_temperature.cool",),
    ),
    _StatusMapping(("get_humidity", "humidity_status"), ("measure_humidity",)),
    _StatusMapping(("get_air_quality", "air_quality", "air_quality_status"), ("measure_airquality",)),
    _StatusMapping(("get_filter_health", "filter_health_status"), ("state_hepafilter_health",)),
    _StatusMapping(("get_fan_speed", "fan_speed_status"), ("state_fanspeed",)),
    _StatusMapping(("get_auto_mode", "auto_mode_status"), ("state_automode",)),
    _StatusMapping(("get_fan_direction", "fan_direction_status"), ("state_fandirection",)),
    _StatusMapping(("get_night_mode", "night_mode_status"), ("state_nightmode",)),
    _StatusMapping(
        ("get_continuous_monitoring", "continuous_monitoring_status"),
        ("state_continuousmonitoring",),
    ),
    _StatusMapping(("get_humidifier_mode", "humidifier_mode_status"), ("state_humidifiermode",)),
    _StatusMapping(("get_target_humidity", "target_humidity_status"), ("target_humidity",)),
    _StatusMapping(("get_oscillation_mode", "oscillation_mode_status"), ("state_oscillationmode",)),
    _StatusMapping(("get_mute_status", "mute_status"), ("volume_mute",)),
    _StatusMapping(("get_thermostat_mode", "thermostat_mode_status"), ("nest_thermostat_mode",)),
    _StatusMapping(("get_eco_status", "eco_status"), ("nest_thermostat_eco",)),
    _StatusMapping(
        ("get_hvac_status", "hvac_status"),
        ("nest_thermostat_hvac", "thermostat_state"),
    ),
    _StatusMapping(("get_curtains_state", "curtains_state"), ("windowcoverings_state",)),
    _StatusMapping(("get_motion_alarm_status", "motion_alarm_status"), ("alarm_motion",)),
    _StatusMapping(("get_tamper_alarm_status", "tamper_alarm_status"), ("alarm_tamper",)),
    _StatusMapping(("get_person_alarm_status", "person_alarm_status"), ("alarm_person",)),
    _StatusMapping(("get_vehicle_alarm_status", "vehicle_alarm_status"), ("alarm_vehicle",)),
    _StatusMapping(("get_pet_alarm_status", "pet_alarm_status"), ("alarm_pet",)),
    _StatusMapping(("get_button_alarm_status", "button_alarm_status"), ("alarm_button",)),
    _StatusMapping(("get_spotlight_status", "spotlight_status"), ("spotlight",)),
    _StatusMapping(("get_ir_lights_status", "ir_lights_status"), ("ir_lights", "infrared")),
    _StatusMapping(("get_day_night_status", "day_night_status"), ("day_night", "day_night_mode")),
)

_WRITE_BY_ACTION = {
    action: mapping
    for mapping in _WRITE_MAPPINGS
    for action in mapping.actions
}
_STATUS_BY_ACTION = {
    action: mapping
    for mapping in _STATUS_MAPPINGS
    for action in mapping.actions
}


class HomeyResolver:
    def __init__(self, catalog: HomeyCatalog, aliases: HomeyAliases):
        self.catalog = catalog
        self.aliases = aliases

    def resolve(
        self,
        *,
        command: str,
        action: str,
        zone: str | None = None,
        device: str | None = None,
        category: str | None = None,
        value_text: str | None = None,
        value_number: int | float | None = None,
        value_bool: bool | None = None,
        capability: str | None = None,
    ) -> HomeyPlan:
        action_id = _action_id(action)
        targets = self._resolve_targets(command, zone, device, category)
        if targets.clarification is not None:
            return HomeyPlan(
                message=targets.clarification,
                operations=(),
                status_reads=(),
                failures=targets.failures,
            )

        # Generic path: the model can write ANY writable capability it sees in the
        # context by passing its exact id, so every device works without a per-action map.
        capability_id = (capability or "").strip()
        if capability_id:
            return self._capability_operation_plan(
                targets,
                capability_id,
                value_text=value_text,
                value_number=value_number,
                value_bool=value_bool,
            )

        status_mapping = _STATUS_BY_ACTION.get(action_id)
        if status_mapping is not None:
            return self._status_plan(targets, status_mapping)

        write_mapping = _WRITE_BY_ACTION.get(action_id)
        if write_mapping is None:
            return HomeyPlan(
                message=f"failed: unsupported action {action_id}",
                operations=(),
                status_reads=(),
                failures=(f"unsupported action {action_id}",),
            )

        return self._operation_plan(
            targets,
            write_mapping,
            value_text=value_text,
            value_number=value_number,
            value_bool=value_bool,
        )

    def _resolve_targets(
        self,
        command: str,
        zone: str | None,
        device: str | None,
        category: str | None,
    ) -> _ResolvedTargets:
        group = self._exact_group(command, device, category)
        if group is not None:
            devices: list[DeviceRecord] = []
            failures: list[str] = []
            for device_name in group.devices:
                matches = self.catalog.find_devices(device_name, group.zone)
                if matches:
                    devices.extend(matches)
                else:
                    failures.append(f"{device_name} not found")
            return _ResolvedTargets(tuple(devices), failures=tuple(failures))

        broad = _is_broad(command, device, category)
        if device and not _is_broad_target(device):
            matches = self._find_devices(device, zone)
            if matches:
                if len(matches) == 1:
                    return _ResolvedTargets(matches)
                return _ResolvedTargets(
                    (),
                    clarification=_device_clarification_message(device, matches),
                )
            return _ResolvedTargets((), failures=(f"{device} not found",))

        if category and not _is_broad_target(category):
            matches = self.catalog.find_category(category, zone, self.aliases)
            if matches:
                return _ResolvedTargets(matches, requires_confirmation=broad)
            return _ResolvedTargets((), failures=(f"{category} not found",))

        if zone:
            zone_devices = self._devices_in_zone(zone)
            if broad:
                return _ResolvedTargets(zone_devices, requires_confirmation=True)
            if zone_devices:
                return _ResolvedTargets(
                    (),
                    clarification=_clarification_message(zone, zone_devices),
                )
            return _ResolvedTargets((), failures=(f"{zone} not found",))

        if broad:
            return _ResolvedTargets(self.catalog.devices, requires_confirmation=True)

        return _ResolvedTargets((), failures=("no matching target",))

    def _exact_group(
        self,
        command: str,
        device: str | None,
        category: str | None,
    ):
        for phrase in (command, device, category):
            group = self.aliases.group_for(phrase)
            if group is not None:
                return group
        return None

    def _find_devices(
        self,
        device_name: str | None,
        zone: str | None,
    ) -> tuple[DeviceRecord, ...]:
        resolved_zone = self.aliases.resolve_zone(zone) if zone else None
        return self.catalog.find_devices(device_name, resolved_zone)

    def _devices_in_zone(self, zone: str) -> tuple[DeviceRecord, ...]:
        normalized_zone = self.aliases.resolve_zone(zone)
        return tuple(
            device
            for device in self.catalog.devices
            if normalize_name(device.zone_name) == normalized_zone
        )

    def _operation_plan(
        self,
        targets: _ResolvedTargets,
        mapping: _ActionMapping,
        *,
        value_text: str | None,
        value_number: int | float | None,
        value_bool: bool | None,
    ) -> HomeyPlan:
        operations: list[HomeyOperation] = []
        failures = list(targets.failures)
        already_done: list[str] = []
        target_count = len(targets.devices)

        for device in targets.devices:
            capability = _first_writable_capability(device, mapping.capability_ids)
            if capability is None:
                existing = _first_capability(device, mapping.capability_ids)
                if existing is None:
                    failures.append(f"{device.name} does not support {mapping.capability_ids[0]}")
                else:
                    failures.append(f"{device.name} cannot write {existing.id}")
                continue

            value = _coerce_value(
                capability,
                mapping,
                value_text=value_text,
                value_number=value_number,
                value_bool=value_bool,
            )
            if isinstance(value, _Failure):
                failures.append(f"{value.message} for {device.name} {capability.id}")
                continue

            high_impact = (
                targets.requires_confirmation
                or mapping.requires_confirmation
                or _target_requires_confirmation(device, capability, target_count)
            )
            operations.append(
                HomeyOperation(
                    device_id=device.id,
                    device_name=device.name,
                    zone_name=device.zone_name,
                    capability_id=capability.id,
                    value=value,
                    value_rule=mapping.value_rule,
                    high_impact=high_impact,
                )
            )

        requires_confirmation = any(operation.high_impact for operation in operations)
        return HomeyPlan(
            message=_operation_message(
                operations,
                already_done,
                failures,
                requires_confirmation=requires_confirmation,
            ),
            operations=tuple(operations),
            status_reads=(),
            requires_confirmation=requires_confirmation,
            failures=tuple(failures),
        )

    def _capability_operation_plan(
        self,
        targets: _ResolvedTargets,
        capability_id: str,
        *,
        value_text: str | None,
        value_number: int | float | None,
        value_bool: bool | None,
    ) -> HomeyPlan:
        operations: list[HomeyOperation] = []
        failures = list(targets.failures)
        target_count = len(targets.devices)

        for device in targets.devices:
            capability = _capability(device, capability_id)
            if capability is None:
                failures.append(f"{device.name} does not support {capability_id}")
                continue
            if not device.can_write(capability.id):
                failures.append(f"{device.name} cannot write {capability.id}")
                continue

            value = _capability_value(
                capability,
                value_text=value_text,
                value_number=value_number,
                value_bool=value_bool,
            )
            if isinstance(value, _Failure):
                failures.append(f"{value.message} for {device.name} {capability.id}")
                continue

            high_impact = (
                targets.requires_confirmation
                or _target_requires_confirmation(device, capability, target_count)
            )
            operations.append(
                HomeyOperation(
                    device_id=device.id,
                    device_name=device.name,
                    zone_name=device.zone_name,
                    capability_id=capability.id,
                    value=value,
                    value_rule="capability",
                    high_impact=high_impact,
                )
            )

        requires_confirmation = any(operation.high_impact for operation in operations)
        return HomeyPlan(
            message=_operation_message(
                operations,
                [],
                failures,
                requires_confirmation=requires_confirmation,
            ),
            operations=tuple(operations),
            status_reads=(),
            requires_confirmation=requires_confirmation,
            failures=tuple(failures),
        )

    def _status_plan(
        self,
        targets: _ResolvedTargets,
        mapping: _StatusMapping,
    ) -> HomeyPlan:
        reads: list[HomeyStatusRead] = []
        failures = list(targets.failures)

        for device in targets.devices:
            capability = _first_readable_capability(device, mapping.capability_ids)
            if capability is None:
                failures.append(f"{device.name} does not expose {mapping.capability_ids[0]}")
                continue
            reads.append(
                HomeyStatusRead(
                    device_id=device.id,
                    device_name=device.name,
                    zone_name=device.zone_name,
                    capability_id=capability.id,
                )
            )

        return HomeyPlan(
            message=_status_message(reads, failures),
            operations=(),
            status_reads=tuple(reads),
            failures=tuple(failures),
        )


@dataclass(frozen=True)
class _Failure:
    message: str


def _action_id(action: str | None) -> str:
    return normalize_name(action).replace(" ", "_").replace("-", "_")


def status_action_for_capability(capability_or_action: str | None) -> str | None:
    """Return a status action for a readable capability ID or status action."""
    action_id = _action_id(capability_or_action)
    if action_id in _STATUS_BY_ACTION:
        return action_id
    for mapping in _STATUS_MAPPINGS:
        for capability_id in mapping.capability_ids:
            if _action_id(capability_id) == action_id:
                return mapping.actions[0]
    return None


def _is_broad(command: str, device: str | None, category: str | None) -> bool:
    text = normalize_name(" ".join(part for part in (command, device, category) if part))
    words = set(text.split())
    return bool(words & _BROAD_WORDS) or any(phrase in text for phrase in _BROAD_PHRASES)


def _is_broad_target(value: str | None) -> bool:
    text = normalize_name(value)
    return text in _BROAD_WORDS or text in _BROAD_PHRASES


def _clarification_message(zone: str, devices: tuple[DeviceRecord, ...]) -> str:
    names = ", ".join(device.name for device in devices)
    return f"clarify: which {zone} device? {names}"


def _device_clarification_message(
    device_name: str,
    devices: tuple[DeviceRecord, ...],
) -> str:
    names = ", ".join(f"{device.name} in {device.zone_name}" for device in devices)
    return f"clarify: which {device_name}? {names}"


def _first_capability(
    device: DeviceRecord,
    capability_ids: tuple[str, ...],
) -> CapabilityRecord | None:
    for capability_id in capability_ids:
        capability = _capability(device, capability_id)
        if capability is not None:
            return capability
    return None


def _first_writable_capability(
    device: DeviceRecord,
    capability_ids: tuple[str, ...],
) -> CapabilityRecord | None:
    for capability_id in capability_ids:
        capability = _capability(device, capability_id)
        if capability is not None and device.can_write(capability.id):
            return capability
    return None


def _first_readable_capability(
    device: DeviceRecord,
    capability_ids: tuple[str, ...],
) -> CapabilityRecord | None:
    for capability_id in capability_ids:
        capability = _capability(device, capability_id)
        if capability is not None and device.can_read(capability.id):
            return capability
    return None


def _capability(device: DeviceRecord, capability_id: str) -> CapabilityRecord | None:
    for capability in device.capabilities:
        if capability.id == capability_id:
            return capability
    return None


def _coerce_value(
    capability: CapabilityRecord,
    mapping: _ActionMapping,
    *,
    value_text: str | None,
    value_number: int | float | None,
    value_bool: bool | None,
) -> Any | _Failure:
    if mapping.value_rule == "literal":
        value = mapping.value
    elif mapping.value_rule == "button":
        value = True
    elif mapping.value_rule == "toggle":
        if not isinstance(capability.value, bool):
            return _Failure("unsupported value")
        value = not capability.value
    else:
        value = _typed_value(
            capability,
            value_text=value_text,
            value_number=value_number,
            value_bool=value_bool,
        )

    if isinstance(value, _Failure):
        return value
    return _validate_value(capability, value)


def _capability_value(
    capability: CapabilityRecord,
    *,
    value_text: str | None,
    value_number: int | float | None,
    value_bool: bool | None,
) -> Any | _Failure:
    """Coerce + validate a value for an arbitrary capability from its catalog metadata."""
    if capability.type == "button":
        return True
    value = _typed_value(
        capability,
        value_text=value_text,
        value_number=value_number,
        value_bool=value_bool,
    )
    if isinstance(value, _Failure):
        return value
    return _validate_value(capability, value)


def _typed_value(
    capability: CapabilityRecord,
    *,
    value_text: str | None,
    value_number: int | float | None,
    value_bool: bool | None,
) -> Any | _Failure:
    if capability.type == "boolean":
        if value_bool is not None:
            return value_bool
        if value_text is not None:
            return _parse_boolean(value_text)
        return _Failure("unsupported value")

    if capability.type == "number":
        if value_number is not None and not isinstance(value_number, bool):
            return value_number
        if value_text is not None:
            return _parse_number(value_text)
        return _Failure("unsupported value")

    if capability.type == "enum":
        if value_text is None:
            return _Failure("unsupported value")
        return _coerce_enum(capability, value_text)

    if capability.type == "button":
        return True

    if value_bool is not None:
        return value_bool
    if value_number is not None and not isinstance(value_number, bool):
        return value_number
    if value_text is not None:
        return value_text
    return _Failure("unsupported value")


def _parse_boolean(value_text: str) -> bool | _Failure:
    normalized = normalize_name(value_text)
    if normalized in _BOOLEAN_TRUE:
        return True
    if normalized in _BOOLEAN_FALSE:
        return False
    return _Failure("unsupported value")


def _parse_number(value_text: str) -> int | float | _Failure:
    try:
        value = float(value_text)
    except ValueError:
        return _Failure("unsupported value")
    if value.is_integer():
        return int(value)
    return value


def _coerce_enum(capability: CapabilityRecord, value_text: str) -> str | _Failure:
    requested = value_text.strip()
    if not requested:
        return _Failure("unsupported value")
    if not capability.enum_value_ids:
        return requested

    normalized_requested = _normalize_enum_value(requested)
    for value_id in capability.enum_value_ids:
        if _normalize_enum_value(value_id) == normalized_requested:
            return value_id
    return _Failure("unsupported value")


def _validate_value(capability: CapabilityRecord, value: Any) -> Any | _Failure:
    if capability.type == "boolean" and not isinstance(value, bool):
        return _Failure("unsupported value")

    if capability.type == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return _Failure("unsupported value")
        if capability.min is not None and value < capability.min:
            return _Failure("value outside supported range")
        if capability.max is not None and value > capability.max:
            return _Failure("value outside supported range")

    if capability.type == "enum":
        if not isinstance(value, str):
            return _Failure("unsupported value")
        if capability.enum_value_ids and value not in capability.enum_value_ids:
            return _Failure("unsupported value")

    if capability.type == "button" and value is not True:
        return _Failure("unsupported value")

    return value


def _normalize_enum_value(value: str) -> str:
    return _VALUE_SEPARATORS_RE.sub(" ", value.strip().casefold())


def _target_requires_confirmation(
    device: DeviceRecord,
    capability: CapabilityRecord,
    target_count: int,
) -> bool:
    class_name = normalize_name(device.class_name)
    if class_name in _HIGH_IMPACT_CLASSES:
        return True
    if capability.id in _THERMOSTAT_CAPABILITIES:
        return True
    if capability.id.startswith("windowcoverings"):
        return True
    if capability.id in _CAMERA_CONTROL_CAPABILITIES:
        return True
    return target_count > 1 and not _safe_light_control(device, capability)


def _safe_light_control(device: DeviceRecord, capability: CapabilityRecord) -> bool:
    return normalize_name(device.class_name) in {"light", "socket"} and capability.id == "onoff"


def _operation_message(
    operations: list[HomeyOperation],
    already_done: list[str],
    failures: list[str],
    *,
    requires_confirmation: bool,
) -> str:
    if operations:
        prefix = "confirm:" if requires_confirmation else "plan:"
        details = ", ".join(_operation_detail(operation) for operation in operations)
        segments = [f"{prefix} {details}"]
        if already_done:
            segments.append("done: " + "; ".join(already_done))
        if failures:
            segments.append("failed: " + "; ".join(failures))
        return "; ".join(segments)

    if already_done and not failures:
        return "done: " + "; ".join(already_done)
    if already_done and failures:
        return "done: " + "; ".join(already_done) + "; failed: " + "; ".join(failures)
    if failures:
        return "failed: " + "; ".join(failures)
    return "failed: no operation planned"


def _operation_detail(operation: HomeyOperation) -> str:
    return f"{operation.device_name} {operation.capability_id}={_format_value(operation.value)}"


def _status_message(reads: list[HomeyStatusRead], failures: list[str]) -> str:
    if reads:
        details = ", ".join(f"{read.device_name} {read.capability_id}" for read in reads)
        if failures:
            return f"status: {details}; failed: {'; '.join(failures)}"
        return f"status: {details}"
    if failures:
        return "failed: " + "; ".join(failures)
    return "failed: no status planned"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
