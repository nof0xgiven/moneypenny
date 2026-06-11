from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .aliases import HomeyAliases, normalize_name


@dataclass(frozen=True)
class CapabilityRecord:
    id: str
    title: str | None
    type: str | None
    value: Any = None
    units: str | None = None
    min: int | float | None = None
    max: int | float | None = None
    enum_value_ids: tuple[str, ...] = ()
    setable: bool | None = None
    writable: bool | None = None
    getable: bool | None = None
    readable: bool | None = None

    @property
    def can_write(self) -> bool:
        return self.setable is True or self.writable is True

    @property
    def can_read(self) -> bool:
        return self.getable is True or self.readable is True


@dataclass(frozen=True)
class DeviceRecord:
    id: str
    name: str
    zone_id: str | None
    zone_name: str
    class_name: str
    capabilities: tuple[CapabilityRecord, ...]

    def has_capability(self, capability_id: str) -> bool:
        return self._capability(capability_id) is not None

    def can_write(self, capability_id: str) -> bool:
        capability = self._capability(capability_id)
        return capability.can_write if capability is not None else False

    def can_read(self, capability_id: str) -> bool:
        capability = self._capability(capability_id)
        return capability.can_read if capability is not None else False

    def _capability(self, capability_id: str) -> CapabilityRecord | None:
        for capability in self.capabilities:
            if capability.id == capability_id:
                return capability
        return None


@dataclass(frozen=True)
class HomeyCatalog:
    devices: tuple[DeviceRecord, ...]

    @classmethod
    def from_payloads(cls, devices: Any, zones: Any) -> "HomeyCatalog":
        zone_names = _zone_names(zones)
        records = [
            _device_record(device_id, device_payload, zone_names)
            for device_id, device_payload in _items(devices)
            if isinstance(device_payload, dict)
        ]
        return cls(
            devices=tuple(
                sorted(
                    records,
                    key=lambda device: (
                        normalize_name(device.zone_name),
                        normalize_name(device.name),
                        device.id,
                    ),
                )
            )
        )

    def find_devices(
        self,
        device_name: str | None,
        zone_name: str | None,
    ) -> tuple[DeviceRecord, ...]:
        normalized_device = normalize_name(device_name)
        normalized_zone = normalize_name(zone_name) if zone_name else None
        return tuple(
            device
            for device in self.devices
            if normalize_name(device.name) == normalized_device
            and (normalized_zone is None or normalize_name(device.zone_name) == normalized_zone)
        )

    def find_category(
        self,
        category_name: str | None,
        zone_name: str | None,
        aliases: HomeyAliases,
    ) -> tuple[DeviceRecord, ...]:
        normalized_zone = aliases.resolve_zone(zone_name) if zone_name else None

        category = aliases.category_for(category_name)
        if category is not None:
            allowed_classes = set(category.classes)
            required_capabilities = set(category.capabilities)
        else:
            # Fallback: treat the category name as a device class so any device type
            # (thermostat, tv, socket, ...) works by category with no alias config.
            allowed_classes = _classes_from_category_name(category_name)
            required_capabilities = set()

        if not allowed_classes:
            return ()

        return tuple(
            device
            for device in self.devices
            if (normalized_zone is None or normalize_name(device.zone_name) == normalized_zone)
            and normalize_name(device.class_name) in allowed_classes
            and all(device.has_capability(capability_id) for capability_id in required_capabilities)
        )


def _classes_from_category_name(category_name: str | None) -> set[str]:
    """Treat a category name as a device class, tolerating simple singular/plural forms."""
    normalized = normalize_name(category_name)
    if not normalized:
        return set()
    classes = {normalized}
    if normalized.endswith("s"):
        classes.add(normalized[:-1])
    else:
        classes.add(normalized + "s")
    return classes


def _items(payload: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(payload, dict):
        return tuple((str(key), value) for key, value in payload.items())
    if isinstance(payload, list):
        return tuple(
            (str(item.get("id", index)), item)
            for index, item in enumerate(payload)
            if isinstance(item, dict)
        )
    return ()


def _zone_names(zones: Any) -> dict[str, str]:
    zone_names: dict[str, str] = {}
    for zone_id, zone_payload in _items(zones):
        if isinstance(zone_payload, dict):
            zone_names[zone_id] = str(zone_payload.get("name") or zone_id)
    return zone_names


def _device_record(device_id: str, payload: dict, zone_names: dict[str, str]) -> DeviceRecord:
    record_id = str(payload.get("id") or device_id)
    zone_id = _optional_str(payload.get("zone") or payload.get("zoneId"))
    return DeviceRecord(
        id=record_id,
        name=str(payload.get("name") or record_id),
        zone_id=zone_id,
        zone_name=zone_names.get(zone_id or "", zone_id or "Unknown"),
        class_name=str(payload.get("class") or payload.get("className") or ""),
        capabilities=_capability_records(payload),
    )


def _capability_records(device_payload: dict) -> tuple[CapabilityRecord, ...]:
    capability_ids = set(
        str(capability_id)
        for capability_id in device_payload.get("capabilities", ())
    )
    capability_objects = device_payload.get("capabilitiesObj", {})
    capability_options = device_payload.get("capabilitiesOptions", {})

    if isinstance(capability_objects, dict):
        capability_ids.update(str(capability_id) for capability_id in capability_objects)
    if isinstance(capability_options, dict):
        capability_ids.update(str(capability_id) for capability_id in capability_options)

    return tuple(
        _capability_record(capability_id, capability_objects, capability_options)
        for capability_id in sorted(capability_ids, key=normalize_name)
    )


def _capability_record(
    capability_id: str,
    capability_objects: Any,
    capability_options: Any,
) -> CapabilityRecord:
    metadata: dict[str, Any] = {}
    if isinstance(capability_options, dict) and isinstance(
        capability_options.get(capability_id),
        dict,
    ):
        metadata.update(capability_options[capability_id])
    if isinstance(capability_objects, dict) and isinstance(
        capability_objects.get(capability_id),
        dict,
    ):
        metadata.update(capability_objects[capability_id])

    return CapabilityRecord(
        id=capability_id,
        title=_optional_str(metadata.get("title")),
        type=_optional_str(metadata.get("type")),
        value=metadata.get("value"),
        units=_optional_str(metadata.get("units")),
        min=metadata.get("min"),
        max=metadata.get("max"),
        enum_value_ids=_enum_value_ids(metadata),
        setable=_bool_or_none(metadata.get("setable")),
        writable=_bool_or_none(metadata.get("writable")),
        getable=_bool_or_none(metadata.get("getable")),
        readable=_bool_or_none(metadata.get("readable")),
    )


def _enum_value_ids(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw_values = metadata.get("values") or metadata.get("enumValues") or ()
    if isinstance(raw_values, dict):
        return tuple(str(value_id) for value_id in raw_values)
    if isinstance(raw_values, list):
        value_ids: list[str] = []
        for raw_value in raw_values:
            if isinstance(raw_value, dict):
                value_id = raw_value.get("id")
            else:
                value_id = raw_value
            if value_id is not None:
                value_ids.append(str(value_id))
        return tuple(value_ids)
    return ()


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
