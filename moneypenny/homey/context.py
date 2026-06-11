from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .aliases import HomeyAliases, HomeyGroup
from .catalog import CapabilityRecord, DeviceRecord, HomeyCatalog


CONFIRMATION_RULE = (
    "Ask for confirmation before thermostat, curtains, camera controls, broad "
    "all/everything commands, or ambiguous broad commands."
)
CONTROL_RULE = (
    "To control ANY device (not just lights), call homey_control with the device and its "
    "exact writable capability id (one marked write below), plus a value matching that "
    "capability's listed type: value_bool for boolean (e.g. onoff), value_number for "
    "number (e.g. target_temperature.cool, dim, state_fanspeed), value_text for enum/"
    "string (use one of the listed choices, e.g. windowcoverings_state=up)."
)


def build_homey_context(catalog: HomeyCatalog, aliases: HomeyAliases) -> str:
    lines = [
        "Homey context:",
        "Use Homey tools only for the devices/capabilities listed here.",
        CONTROL_RULE,
        CONFIRMATION_RULE,
        "Available devices by zone:",
    ]
    lines.extend(_zone_lines(catalog.devices))
    lines.append("Categories:")
    lines.extend(_category_lines(aliases))
    lines.append("Alias groups:")
    lines.extend(_group_lines(aliases.groups))
    return "\n".join(line for line in lines if line)


def _zone_lines(devices: Iterable[DeviceRecord]) -> list[str]:
    devices_by_zone: dict[str, list[DeviceRecord]] = defaultdict(list)
    for device in devices:
        devices_by_zone[device.zone_name].append(device)

    lines: list[str] = []
    for zone_name in sorted(devices_by_zone):
        device_parts = []
        for device in devices_by_zone[zone_name]:
            capability_text = ", ".join(
                _capability_text(capability)
                for capability in device.capabilities
            )
            device_parts.append(f"{device.name} ({device.class_name}; {capability_text})")
        lines.append(f"- {zone_name}: {'; '.join(device_parts)}")
    return lines


def _capability_text(capability: CapabilityRecord) -> str:
    text = capability.id
    if capability.value is not None:
        text = f"{text}={_format_value(capability.value)}{capability.units or ''}"

    access = []
    if capability.can_read:
        access.append("read")
    if capability.can_write:
        access.append("write")
    if access:
        metadata = _capability_metadata(capability)
        if metadata:
            text = f"{text} [{'; '.join(metadata)}]"
        text = f"{text} ({'/'.join(access)})"
    return text


def _capability_metadata(capability: CapabilityRecord) -> list[str]:
    metadata = []
    if capability.title:
        metadata.append(capability.title)
    if capability.type:
        metadata.append(capability.type)

    range_text = _range_text(capability)
    if range_text:
        metadata.append(range_text)

    if capability.enum_value_ids:
        metadata.append(f"choices {'/'.join(capability.enum_value_ids)}")
    return metadata


def _range_text(capability: CapabilityRecord) -> str | None:
    if capability.min is None and capability.max is None:
        return None

    minimum = "" if capability.min is None else _format_value(capability.min)
    maximum = "" if capability.max is None else _format_value(capability.max)
    return f"range {minimum}..{maximum}{capability.units or ''}"


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _category_lines(aliases: HomeyAliases) -> list[str]:
    lines: list[str] = []
    for category in aliases.categories.values():
        labels = ", ".join(category.labels) or "none"
        classes = ", ".join(category.classes) or "any"
        capabilities = ", ".join(category.capabilities) or "any"
        lines.append(
            f"- {category.name} labels {labels} classes {classes} "
            f"capabilities {capabilities}"
        )
    return lines


def _group_lines(groups: Iterable[HomeyGroup]) -> list[str]:
    lines: list[str] = []
    for group in groups:
        aliases = ", ".join(group.aliases) or "none"
        devices = ", ".join(group.devices)
        lines.append(
            f"- {group.name} aliases {aliases}; zone {group.zone}; "
            f"devices {devices}; capability {group.capability}"
        )
        lines.append(
            f"{_sentence_name(group.name)} means {_and_join(group.devices)} "
            f"in {group.zone}."
        )
    return lines


def _sentence_name(name: str) -> str:
    return f"{name[:1].upper()}{name[1:]}"


def _and_join(values: tuple[str, ...]) -> str:
    if len(values) <= 1:
        return "".join(values)
    return f"{', '.join(values[:-1])} and {values[-1]}"
