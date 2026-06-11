from __future__ import annotations

import math

import pytest

from moneypenny.homey.aliases import HomeyAliases, HomeyCategory, HomeyGroup
from moneypenny.homey.catalog import CapabilityRecord, DeviceRecord, HomeyCatalog
from moneypenny.homey.resolver import HomeyResolver


def _capability(
    capability_id: str,
    capability_type: str,
    *,
    value=None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    enum_value_ids: tuple[str, ...] = (),
    writable: bool | None = None,
    readable: bool | None = None,
) -> CapabilityRecord:
    return CapabilityRecord(
        id=capability_id,
        title=capability_id,
        type=capability_type,
        value=value,
        min=minimum,
        max=maximum,
        enum_value_ids=enum_value_ids,
        writable=writable,
        setable=writable,
        readable=readable,
        getable=readable,
    )


def _device(
    device_id: str,
    name: str,
    zone_name: str,
    class_name: str,
    *capabilities: CapabilityRecord,
) -> DeviceRecord:
    return DeviceRecord(
        id=device_id,
        name=name,
        zone_id=f"zone-{zone_name.casefold().replace(' ', '-')}",
        zone_name=zone_name,
        class_name=class_name,
        capabilities=capabilities,
    )


def _resolver(
    *,
    ceiling_on: bool = False,
    include_read_only_group: bool = False,
) -> HomeyResolver:
    devices = [
        _device(
            "device-arches",
            "Arches",
            "Living Room",
            "socket",
            _capability("onoff", "boolean", value=True, writable=True, readable=True),
        ),
        _device(
            "device-ceiling",
            "Ceiling",
            "Living Room",
            "socket",
            _capability("onoff", "boolean", value=ceiling_on, writable=True, readable=True),
        ),
        _device(
            "device-lounge",
            "Lounge",
            "Living Room",
            "fan",
            _capability("onoff", "boolean", value=True, writable=True, readable=True),
            _capability(
                "state_fanspeed",
                "enum",
                value="medium",
                enum_value_ids=("low", "medium", "high"),
                writable=True,
                readable=True,
            ),
            _capability("state_automode", "boolean", value=False, writable=True, readable=True),
            _capability("state_nightmode", "boolean", value=False, writable=True, readable=True),
            _capability(
                "target_humidity",
                "number",
                value=50,
                minimum=40,
                maximum=70,
                writable=True,
            ),
            _capability("measure_airquality", "enum", value="good", readable=True),
        ),
        _device(
            "device-sony-tv",
            "Sony TV",
            "Living Room",
            "tv",
            _capability("onoff", "boolean", value=False, writable=True, readable=True),
            _capability("volume_mute", "button", writable=True),
            _capability("volume_up", "button", writable=True),
            _capability("volume_down", "button", writable=True),
            _capability("channel_up", "button", writable=True),
            _capability("channel_down", "button", writable=True),
        ),
        _device(
            "device-thermostat",
            "Thermostat",
            "Living Room",
            "thermostat",
            _capability("measure_temperature", "number", value=22.5, readable=True),
            _capability("measure_humidity", "number", value=44, readable=True),
            _capability(
                "nest_thermostat_hvac",
                "enum",
                value="cooling",
                enum_value_ids=("idle", "cooling", "heating"),
                readable=True,
            ),
            _capability(
                "target_temperature.cool",
                "number",
                value=21,
                minimum=16,
                maximum=30,
                writable=True,
            ),
            _capability(
                "nest_thermostat_mode",
                "enum",
                value="cool",
                enum_value_ids=("heat", "cool", "heat-cool", "off"),
                writable=True,
            ),
            _capability("nest_thermostat_eco", "boolean", value=False, writable=True),
        ),
        _device(
            "device-curtains",
            "Curtains",
            "Master Bedroom",
            "windowcoverings",
            _capability(
                "windowcoverings_state",
                "enum",
                value="idle",
                enum_value_ids=("up", "down", "idle"),
                writable=True,
                readable=True,
            ),
            _capability("windowcoverings_tilt_up", "button", writable=True),
            _capability("windowcoverings_tilt_down", "button", writable=True),
        ),
        _device(
            "device-boys-camera",
            "Boys",
            "Hallway",
            "camera",
            _capability("alarm_motion", "boolean", value=False, readable=True),
            _capability("alarm_tamper", "boolean", value=False, readable=True),
            _capability("alarm_person", "boolean", value=False, readable=True),
            _capability("alarm_vehicle", "boolean", value=False, readable=True),
            _capability("alarm_pet", "boolean", value=False, readable=True),
            _capability("alarm_button", "boolean", value=False, readable=True),
            _capability("ptz_up", "button", writable=True),
            _capability("ptz_down", "button", writable=True),
            _capability("ptz_left", "button", writable=True),
            _capability("ptz_right", "button", writable=True),
            _capability("zoom_in", "button", writable=True),
            _capability("zoom_out", "button", writable=True),
            _capability("spotlight", "boolean", value=False, writable=True, readable=True),
            _capability("ir_lights", "boolean", value=True, writable=True, readable=True),
            _capability(
                "day_night",
                "enum",
                value="auto",
                enum_value_ids=("auto", "day", "night"),
                writable=True,
                readable=True,
            ),
        ),
        _device(
            "device-read-only-lamp",
            "Read Only Lamp",
            "Living Room",
            "sensor",
            _capability("onoff", "boolean", value=True, readable=True),
        ),
    ]
    groups = [
        HomeyGroup(
            name="living room lights",
            aliases=("lights in the living room", "the living room lights"),
            zone="Living Room",
            devices=("Arches", "Ceiling"),
            capability="onoff",
        )
    ]
    if include_read_only_group:
        groups.append(
            HomeyGroup(
                name="mixed living lights",
                aliases=(),
                zone="Living Room",
                devices=("Arches", "Read Only Lamp"),
                capability="onoff",
            )
        )

    aliases = HomeyAliases(
        zones={
            "living room": ("living room",),
            "lounge": ("living room",),
            "master bedroom": ("master bedroom",),
            "hallway": ("hallway",),
        },
        groups=tuple(groups),
        categories={
            "lights": HomeyCategory(
                name="lights",
                classes=("socket",),
                capabilities=("onoff",),
                labels=("lights", "lamps"),
            ),
            "fans": HomeyCategory(
                name="fans",
                classes=("fan",),
                capabilities=("onoff",),
                labels=("fan", "fans"),
            ),
            "cameras": HomeyCategory(
                name="cameras",
                classes=("camera",),
                capabilities=(),
                labels=("camera", "cameras"),
            ),
        },
    )
    return HomeyResolver(HomeyCatalog(tuple(devices)), aliases)


def _resolver_for_devices(*devices: DeviceRecord) -> HomeyResolver:
    return HomeyResolver(HomeyCatalog(tuple(devices)), HomeyAliases.empty())


def _resolve(resolver: HomeyResolver, **kwargs):
    return resolver.resolve(
        command=kwargs.pop("command", ""),
        action=kwargs.pop("action", ""),
        **kwargs,
    )


def _only_operation(plan):
    assert len(plan.operations) == 1
    return plan.operations[0]


def _only_status_read(plan):
    assert plan.operations == ()
    assert len(plan.status_reads) == 1
    return plan.status_reads[0]


def test_exact_device_low_risk_control_needs_no_confirmation():
    plan = _resolve(
        _resolver(ceiling_on=False),
        command="turn on Ceiling in Living Room",
        zone="Living Room",
        device="Ceiling",
        action="turn_on",
    )

    operation = _only_operation(plan)
    assert operation.device_name == "Ceiling"
    assert operation.zone_name == "Living Room"
    assert operation.capability_id == "onoff"
    assert operation.value is True
    assert operation.high_impact is False
    assert plan.requires_confirmation is False
    assert plan.status_reads == ()


def test_exact_device_with_zone_alias_resolves_canonical_zone():
    plan = _resolve(
        _resolver(ceiling_on=False),
        command="turn on Ceiling in the lounge",
        zone="lounge",
        device="Ceiling",
        action="turn_on",
    )

    operation = _only_operation(plan)
    assert operation.device_name == "Ceiling"
    assert operation.zone_name == "Living Room"
    assert operation.capability_id == "onoff"
    assert operation.value is True
    assert plan.requires_confirmation is False


def test_duplicate_exact_device_name_without_zone_returns_dynamic_clarification():
    resolver = _resolver_for_devices(
        _device(
            "device-kitchen-lamp",
            "Lamp",
            "Kitchen",
            "socket",
            _capability("onoff", "boolean", value=False, writable=True, readable=True),
        ),
        _device(
            "device-hallway-lamp",
            "Lamp",
            "Hallway",
            "socket",
            _capability("onoff", "boolean", value=False, writable=True, readable=True),
        ),
    )

    plan = _resolve(
        resolver,
        command="turn on Lamp",
        device="Lamp",
        action="turn_on",
    )

    assert plan.operations == ()
    assert plan.status_reads == ()
    assert plan.message.startswith("clarify:")
    assert "Lamp" in plan.message
    assert "Kitchen" in plan.message
    assert "Hallway" in plan.message


def test_category_fan_out_targets_matching_living_room_lights_only():
    plan = _resolve(
        _resolver(ceiling_on=True),
        command="turn off living room lights",
        zone="Living Room",
        category="lights",
        action="turn_off",
    )

    assert [(op.device_name, op.capability_id, op.value) for op in plan.operations] == [
        ("Arches", "onoff", False),
        ("Ceiling", "onoff", False),
    ]
    assert plan.requires_confirmation is False
    assert "Read Only Lamp" not in plan.message


def test_already_state_plans_operation_for_live_execution_check():
    plan = _resolve(
        _resolver(ceiling_on=True),
        command="turn on Ceiling",
        zone="Living Room",
        device="Ceiling",
        action="turn_on",
    )

    operation = _only_operation(plan)
    assert operation.capability_id == "onoff"
    assert operation.value is True
    assert plan.requires_confirmation is False


def test_toggle_resolves_from_current_boolean_state():
    plan = _resolve(
        _resolver(),
        command="toggle Arches",
        zone="Living Room",
        device="Arches",
        action="toggle",
    )

    operation = _only_operation(plan)
    assert operation.capability_id == "onoff"
    assert operation.value is False


def test_zone_only_command_returns_dynamic_clarification_from_matching_devices():
    plan = _resolve(
        _resolver(),
        command="turn off living room",
        zone="Living Room",
        action="turn_off",
    )

    assert plan.operations == ()
    assert plan.status_reads == ()
    assert plan.message.startswith("clarify:")
    assert "Living Room" in plan.message
    assert "Arches" in plan.message
    assert "Sony TV" in plan.message
    assert "Thermostat" in plan.message
    assert "Arches/Ceiling" not in plan.message


def test_broad_everything_command_requires_confirmation_before_writes():
    plan = _resolve(
        _resolver(),
        command="turn off everything",
        action="turn_off",
    )

    assert plan.requires_confirmation is True
    assert plan.operations
    assert all(operation.high_impact is True for operation in plan.operations)
    assert plan.message.startswith("confirm:")


@pytest.mark.parametrize("category", ["all", "everything"])
@pytest.mark.parametrize("zone", [None, "Living Room"])
def test_broad_category_targets_require_confirmation_before_writes(
    category: str,
    zone: str | None,
):
    plan = _resolve(
        _resolver(ceiling_on=True),
        command=f"turn off {category}",
        zone=zone,
        category=category,
        action="turn_off",
    )

    assert plan.requires_confirmation is True
    assert plan.operations
    assert all(operation.high_impact is True for operation in plan.operations)
    assert plan.message.startswith("confirm:")
    assert f"{category} not found" not in plan.message
    if zone is not None:
        assert {operation.zone_name for operation in plan.operations} == {zone}


def test_multiple_non_light_category_targets_require_confirmation():
    aliases = HomeyAliases(
        zones={"living room": ("living room",)},
        groups=(),
        categories={
            "fans": HomeyCategory(
                name="fans",
                classes=("fan",),
                capabilities=("onoff",),
                labels=("fans",),
            )
        },
    )
    resolver = HomeyResolver(
        HomeyCatalog(
            (
                _device(
                    "fan-1",
                    "Lounge",
                    "Living Room",
                    "fan",
                    _capability("onoff", "boolean", value=True, writable=True, readable=True),
                ),
                _device(
                    "fan-2",
                    "Dining Fan",
                    "Living Room",
                    "fan",
                    _capability("onoff", "boolean", value=True, writable=True, readable=True),
                ),
            )
        ),
        aliases,
    )

    plan = _resolve(
        resolver,
        command="turn off the fans",
        zone="Living Room",
        category="fans",
        action="turn_off",
    )

    assert {
        (operation.device_name, operation.high_impact)
        for operation in plan.operations
    } == {
        ("Dining Fan", True),
        ("Lounge", True),
    }
    assert plan.requires_confirmation is True
    assert plan.message.startswith("confirm:")


def test_read_only_capability_cannot_be_controlled():
    plan = _resolve(
        _resolver(),
        command="turn off Read Only Lamp",
        zone="Living Room",
        device="Read Only Lamp",
        action="turn_off",
    )

    assert plan.operations == ()
    assert plan.message.startswith("failed:")
    assert "Read Only Lamp" in plan.message
    assert "onoff" in plan.message


def test_invalid_numeric_value_outside_range_fails():
    plan = _resolve(
        _resolver(),
        command="set Lounge target humidity to 90",
        zone="Living Room",
        device="Lounge",
        action="set_target_humidity",
        value_number=90,
    )

    assert plan.operations == ()
    assert plan.message.startswith("failed: value outside supported range")
    assert "Lounge" in plan.message


@pytest.mark.parametrize("value_number", [math.nan, math.inf, -math.inf])
def test_invalid_non_finite_numeric_values_fail(value_number: float):
    plan = _resolve(
        _resolver(),
        command="set Lounge target humidity",
        zone="Living Room",
        device="Lounge",
        action="set_target_humidity",
        value_number=value_number,
    )

    assert plan.operations == ()
    assert plan.message.startswith("failed:")
    assert "unsupported value" in plan.message


def test_invalid_enum_value_fails():
    plan = _resolve(
        _resolver(),
        command="set Lounge fan speed to turbo",
        zone="Living Room",
        device="Lounge",
        action="set_fan_speed",
        value_text="turbo",
    )

    assert plan.operations == ()
    assert plan.message.startswith("failed: unsupported value")
    assert "state_fanspeed" in plan.message


def test_fan_speed_maps_to_writable_state_fanspeed():
    plan = _resolve(
        _resolver(),
        command="set Lounge fan speed to high",
        zone="Living Room",
        device="Lounge",
        action="set_fan_speed",
        value_text="high",
    )

    operation = _only_operation(plan)
    assert operation.device_name == "Lounge"
    assert operation.capability_id == "state_fanspeed"
    assert operation.value == "high"
    assert plan.requires_confirmation is False


@pytest.mark.parametrize(
    ("action", "value_bool", "capability_id"),
    [
        ("set_night_mode", True, "state_nightmode"),
        ("set_auto_mode", True, "state_automode"),
    ],
)
def test_fan_mode_controls_map_to_writable_fan_state_capabilities(
    action: str,
    value_bool: bool,
    capability_id: str,
):
    plan = _resolve(
        _resolver(),
        command=f"{action} Lounge",
        zone="Living Room",
        device="Lounge",
        action=action,
        value_bool=value_bool,
    )

    operation = _only_operation(plan)
    assert operation.capability_id == capability_id
    assert operation.value is value_bool


def test_fan_air_quality_capability_alias_reads_measure_airquality():
    plan = _resolve(
        _resolver(),
        command="what is the living room air quality",
        zone="Living Room",
        device="Lounge",
        action="air_quality",
    )

    status_read = _only_status_read(plan)
    assert status_read.device_name == "Lounge"
    assert status_read.capability_id == "measure_airquality"


def test_status_read_fails_when_capability_has_value_but_is_not_readable():
    resolver = _resolver_for_devices(
        _device(
            "device-valued-lamp",
            "Valued Lamp",
            "Living Room",
            "socket",
            _capability("onoff", "boolean", value=True, readable=False),
        )
    )

    plan = _resolve(
        resolver,
        command="is Valued Lamp on",
        zone="Living Room",
        device="Valued Lamp",
        action="get_power_status",
    )

    assert plan.operations == ()
    assert plan.status_reads == ()
    assert plan.message.startswith("failed:")
    assert "Valued Lamp" in plan.message


def test_control_uses_writable_fallback_capability_when_primary_exists_unwritable():
    resolver = _resolver_for_devices(
        _device(
            "device-driveway-camera",
            "Driveway Camera",
            "Hallway",
            "camera",
            _capability("ir_lights", "boolean", value=True, writable=False, readable=True),
            _capability("infrared", "boolean", value=True, writable=True, readable=True),
        )
    )

    plan = _resolve(
        resolver,
        command="turn off Driveway Camera infrared",
        zone="Hallway",
        device="Driveway Camera",
        action="ir_lights",
        value_bool=False,
    )

    operation = _only_operation(plan)
    assert operation.capability_id == "infrared"
    assert operation.value is False


def test_status_uses_readable_fallback_capability_when_primary_exists_unreadable():
    resolver = _resolver_for_devices(
        _device(
            "device-driveway-camera",
            "Driveway Camera",
            "Hallway",
            "camera",
            _capability("ir_lights", "boolean", value=True, readable=False),
            _capability("infrared", "boolean", value=True, readable=True),
        )
    )

    plan = _resolve(
        resolver,
        command="are Driveway Camera infrared lights on",
        zone="Hallway",
        device="Driveway Camera",
        action="get_ir_lights_status",
    )

    status_read = _only_status_read(plan)
    assert status_read.capability_id == "infrared"


@pytest.mark.parametrize(
    ("action", "capability_id"),
    [
        ("mute", "volume_mute"),
        ("volume_up", "volume_up"),
        ("volume_down", "volume_down"),
        ("channel_up", "channel_up"),
        ("channel_down", "channel_down"),
    ],
)
def test_tv_mute_volume_and_channel_actions_map_to_button_writes(
    action: str,
    capability_id: str,
):
    plan = _resolve(
        _resolver(),
        command=f"{action} Sony TV",
        zone="Living Room",
        device="Sony TV",
        action=action,
    )

    operation = _only_operation(plan)
    assert operation.capability_id == capability_id
    assert operation.value is True
    assert plan.requires_confirmation is False


@pytest.mark.parametrize(
    ("action", "value", "expected_capability", "expected_value"),
    [
        ("set_temperature", 23, "target_temperature.cool", 23),
        ("set_thermostat_mode", "heat", "nest_thermostat_mode", "heat"),
        ("set_eco", True, "nest_thermostat_eco", True),
    ],
)
def test_thermostat_changes_require_confirmation(
    action: str,
    value,
    expected_capability: str,
    expected_value,
):
    kwargs = {"value_number": value} if isinstance(value, (int, float)) else {}
    kwargs.update({"value_bool": value} if isinstance(value, bool) else {})
    kwargs.update({"value_text": value} if isinstance(value, str) else {})

    plan = _resolve(
        _resolver(),
        command=f"{action} Thermostat",
        zone="Living Room",
        device="Thermostat",
        action=action,
        **kwargs,
    )

    operation = _only_operation(plan)
    assert operation.capability_id == expected_capability
    assert operation.value == expected_value
    assert operation.high_impact is True
    assert plan.requires_confirmation is True
    assert plan.message.startswith("confirm:")


@pytest.mark.parametrize(
    ("action", "capability_id"),
    [
        ("get_temperature", "measure_temperature"),
        ("get_humidity", "measure_humidity"),
        ("get_hvac_status", "nest_thermostat_hvac"),
    ],
)
def test_thermostat_status_reads_use_readable_capabilities(action: str, capability_id: str):
    plan = _resolve(
        _resolver(),
        command=f"{action} Thermostat",
        zone="Living Room",
        device="Thermostat",
        action=action,
    )

    status_read = _only_status_read(plan)
    assert status_read.capability_id == capability_id
    assert plan.requires_confirmation is False


@pytest.mark.parametrize(("action", "expected_value"), [("open", "up"), ("close", "down")])
def test_curtains_open_close_map_to_state_and_require_confirmation(
    action: str,
    expected_value: str,
):
    plan = _resolve(
        _resolver(),
        command=f"{action} Curtains",
        zone="Master Bedroom",
        device="Curtains",
        action=action,
    )

    operation = _only_operation(plan)
    assert operation.capability_id == "windowcoverings_state"
    assert operation.value == expected_value
    assert operation.high_impact is True
    assert plan.requires_confirmation is True


@pytest.mark.parametrize(
    ("action", "capability_id"),
    [("tilt_up", "windowcoverings_tilt_up"), ("tilt_down", "windowcoverings_tilt_down")],
)
def test_curtains_tilt_maps_to_button_and_requires_confirmation(
    action: str,
    capability_id: str,
):
    plan = _resolve(
        _resolver(),
        command=f"{action} Curtains",
        zone="Master Bedroom",
        device="Curtains",
        action=action,
    )

    operation = _only_operation(plan)
    assert operation.capability_id == capability_id
    assert operation.value is True
    assert operation.high_impact is True
    assert plan.requires_confirmation is True


def test_camera_alarm_status_is_read_only_status_plan():
    plan = _resolve(
        _resolver(),
        command="is there motion on Boys camera",
        zone="Hallway",
        device="Boys",
        action="get_motion_alarm_status",
    )

    status_read = _only_status_read(plan)
    assert status_read.capability_id == "alarm_motion"
    assert plan.requires_confirmation is False


def test_camera_tamper_alarm_status_is_read_only_status_plan():
    plan = _resolve(
        _resolver(),
        command="is Boys camera tampered",
        zone="Hallway",
        device="Boys",
        action="get_tamper_alarm_status",
    )

    status_read = _only_status_read(plan)
    assert status_read.capability_id == "alarm_tamper"
    assert plan.requires_confirmation is False


def test_camera_power_control_requires_confirmation_by_device_class():
    resolver = _resolver_for_devices(
        _device(
            "device-boys-camera",
            "Boys",
            "Hallway",
            "camera",
            _capability("onoff", "boolean", value=True, writable=True, readable=True),
        )
    )

    plan = _resolve(
        resolver,
        command="turn off Boys camera",
        zone="Hallway",
        device="Boys",
        action="turn_off",
    )

    operation = _only_operation(plan)
    assert operation.capability_id == "onoff"
    assert operation.high_impact is True
    assert plan.requires_confirmation is True
    assert plan.message.startswith("confirm:")


@pytest.mark.parametrize(
    ("action", "capability_id", "kwargs", "expected_value"),
    [
        ("ptz_up", "ptz_up", {}, True),
        ("ptz_down", "ptz_down", {}, True),
        ("ptz_left", "ptz_left", {}, True),
        ("ptz_right", "ptz_right", {}, True),
        ("zoom_in", "zoom_in", {}, True),
        ("zoom_out", "zoom_out", {}, True),
        ("spotlight", "spotlight", {"value_bool": True}, True),
        ("ir_lights", "ir_lights", {"value_bool": False}, False),
        ("day_night", "day_night", {"value_text": "night"}, "night"),
    ],
)
def test_camera_controls_require_confirmation(
    action: str,
    capability_id: str,
    kwargs: dict,
    expected_value,
):
    plan = _resolve(
        _resolver(),
        command=f"{action} Boys camera",
        zone="Hallway",
        device="Boys",
        action=action,
        **kwargs,
    )

    operation = _only_operation(plan)
    assert operation.capability_id == capability_id
    assert operation.value == expected_value
    assert operation.high_impact is True
    assert plan.requires_confirmation is True


def test_grouped_operations_include_partial_failure_messages_for_unsupported_targets():
    plan = _resolve(
        _resolver(include_read_only_group=True),
        command="mixed living lights",
        action="turn_off",
    )

    assert [(operation.device_name, operation.value) for operation in plan.operations] == [
        ("Arches", False)
    ]
    assert plan.failures
    assert "Read Only Lamp" in plan.message
    assert "failed:" in plan.message


# --- generic capability path (control any writable capability by id) -----------------


def test_generic_capability_sets_writable_number_with_confirmation():
    plan = _resolve(
        _resolver(),
        command="set the thermostat to 22",
        device="Thermostat",
        zone="Living Room",
        capability="target_temperature.cool",
        value_number=22,
    )
    operation = _only_operation(plan)
    assert operation.capability_id == "target_temperature.cool"
    assert operation.value == 22
    assert plan.requires_confirmation is True  # thermostat class is high-impact


def test_generic_capability_boolean_via_value_bool():
    plan = _resolve(
        _resolver(),
        command="turn the lounge fan off",
        device="Lounge",
        capability="onoff",
        value_bool=False,
    )
    operation = _only_operation(plan)
    assert operation.capability_id == "onoff"
    assert operation.value is False
    assert plan.requires_confirmation is False  # single non-light device, low risk


def test_generic_capability_enum_via_value_text():
    plan = _resolve(
        _resolver(),
        command="set the lounge fan to high",
        device="Lounge",
        capability="state_fanspeed",
        value_text="high",
    )
    operation = _only_operation(plan)
    assert operation.capability_id == "state_fanspeed"
    assert operation.value == "high"


def test_generic_capability_missing_on_device_fails():
    plan = _resolve(
        _resolver(),
        command="dim the ceiling",
        device="Ceiling",
        capability="dim",
        value_number=0.5,
    )
    assert plan.operations == ()
    assert any("does not support dim" in failure for failure in plan.failures)


def test_generic_capability_read_only_cannot_be_written():
    plan = _resolve(
        _resolver(),
        command="set thermostat measured temperature",
        device="Thermostat",
        capability="measure_temperature",
        value_number=20,
    )
    assert plan.operations == ()
    assert any("cannot write measure_temperature" in failure for failure in plan.failures)


def test_generic_capability_value_out_of_range_fails():
    plan = _resolve(
        _resolver(),
        command="set the thermostat to 99",
        device="Thermostat",
        capability="target_temperature.cool",
        value_number=99,
    )
    assert plan.operations == ()
    assert any("outside supported range" in failure for failure in plan.failures)


# --- category resolution for non-light device classes --------------------------------


def _thermostat_device() -> DeviceRecord:
    return _device(
        "device-kitchen-thermostat",
        "Thermostat",
        "Kitchen",
        "thermostat",
        _capability("measure_temperature", "number", value=23.7, readable=True),
        _capability(
            "target_temperature.cool",
            "number",
            value=23,
            minimum=9,
            maximum=32,
            writable=True,
            readable=True,
        ),
    )


def test_category_falls_back_to_device_class_without_alias():
    # No alias defines "thermostat"; it should resolve by matching the device class.
    resolver = _resolver_for_devices(_thermostat_device())
    plan = _resolve(
        resolver,
        command="what's the kitchen temperature",
        action="get_temperature",
        category="thermostat",
        zone="Kitchen",
    )
    read = _only_status_read(plan)
    assert read.device_name == "Thermostat"
    assert read.capability_id == "measure_temperature"


def test_category_class_fallback_allows_capability_write():
    resolver = _resolver_for_devices(_thermostat_device())
    plan = _resolve(
        resolver,
        command="set the kitchen thermostat to 21",
        category="thermostat",
        zone="Kitchen",
        capability="target_temperature.cool",
        value_number=21,
    )
    operation = _only_operation(plan)
    assert operation.capability_id == "target_temperature.cool"
    assert operation.value == 21
    assert plan.requires_confirmation is True  # thermostat is high-impact


def test_alias_file_defines_non_light_categories():
    aliases = HomeyAliases.load()
    assert aliases.category_for("thermostat") is not None
    assert aliases.category_for("curtains") is not None
    assert aliases.category_for("tv") is not None
    assert aliases.category_for("sockets") is not None
