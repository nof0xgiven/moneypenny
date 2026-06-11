from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ALIASES_PATH = Path(__file__).resolve().parent.parent.parent / "homey-aliases.json"

_SPACE_RE = re.compile(r"\s+")


def normalize_name(name: str | None) -> str:
    return _SPACE_RE.sub(" ", str(name or "").strip().casefold())


@dataclass(frozen=True)
class HomeyGroup:
    name: str
    zone: str
    devices: tuple[str, ...]
    capability: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class HomeyCategory:
    name: str
    classes: tuple[str, ...]
    capabilities: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class HomeyAliases:
    zones: dict[str, tuple[str, ...]]
    groups: tuple[HomeyGroup, ...]
    categories: dict[str, HomeyCategory]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_ALIASES_PATH) -> "HomeyAliases":
        alias_path = Path(path)
        if not alias_path.exists():
            return cls.empty()

        with alias_path.open("r", encoding="utf-8") as alias_file:
            payload = json.load(alias_file)

        return cls(
            zones=_load_zones(payload.get("zones", {})),
            groups=_load_groups(payload.get("groups", ())),
            categories=_load_categories(payload.get("categories", {})),
        )

    @classmethod
    def empty(cls) -> "HomeyAliases":
        return cls(zones={}, groups=(), categories={})

    def resolve_zone(self, name: str | None) -> str:
        normalized = normalize_name(name)
        matches = self.zones.get(normalized)
        if not matches:
            return normalized
        return matches[0]

    def group_for(self, phrase: str | None) -> HomeyGroup | None:
        normalized = normalize_name(phrase)
        for group in self.groups:
            names = (group.name, *group.aliases)
            if normalized in {normalize_name(name) for name in names}:
                return group
        return None

    def category_for(self, phrase: str | None) -> HomeyCategory | None:
        normalized = normalize_name(phrase)
        category = self.categories.get(normalized)
        if category is not None:
            return category

        for category in self.categories.values():
            if normalized in {normalize_name(label) for label in category.labels}:
                return category
        return None


def _load_zones(raw_zones: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw_zones, dict):
        return {}
    zones: dict[str, tuple[str, ...]] = {}
    for alias, canonical_names in raw_zones.items():
        if isinstance(canonical_names, str):
            canonical_values = (normalize_name(canonical_names),)
        else:
            canonical_values = tuple(
                normalize_name(zone_name)
                for zone_name in canonical_names
                if normalize_name(zone_name)
            )
        zones[normalize_name(alias)] = canonical_values
    return zones


def _load_groups(raw_groups: Any) -> tuple[HomeyGroup, ...]:
    if not isinstance(raw_groups, list):
        return ()

    groups: list[HomeyGroup] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue
        groups.append(
            HomeyGroup(
                name=str(raw_group.get("name", "")),
                aliases=tuple(str(alias) for alias in raw_group.get("aliases", ())),
                zone=str(raw_group.get("zone", "")),
                devices=tuple(str(device) for device in raw_group.get("devices", ())),
                capability=str(raw_group.get("capability", "")),
            )
        )
    return tuple(sorted(groups, key=lambda group: normalize_name(group.name)))


def _load_categories(raw_categories: Any) -> dict[str, HomeyCategory]:
    if not isinstance(raw_categories, dict):
        return {}

    categories: dict[str, HomeyCategory] = {}
    for name, raw_category in raw_categories.items():
        if not isinstance(raw_category, dict):
            continue
        category_name = str(name)
        categories[normalize_name(category_name)] = HomeyCategory(
            name=category_name,
            classes=tuple(
                normalize_name(class_name)
                for class_name in raw_category.get("classes", ())
            ),
            capabilities=tuple(
                str(capability)
                for capability in raw_category.get("capabilities", ())
            ),
            labels=tuple(normalize_name(label) for label in raw_category.get("labels", ())),
        )
    return dict(sorted(categories.items()))
