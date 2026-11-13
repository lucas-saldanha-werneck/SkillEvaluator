# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the deterministic plugin quality score.

The packaged ``plugin-quality-default.yaml`` lists every rule with its dimension,
severity, points, scope and source. A custom file names any subset of rules and
keys; omitted keys keep the packaged values, lists replace (they do not merge),
and ``points: 0`` turns a rule off. ``dimension`` and ``applies_to`` are fixed.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal, TypedDict

import yaml

from skillevaluator.quality_config import (
    QualityConfigError,
    QualityConfigProvenance,
    _reject_unknown_keys,
    _UniqueKeySafeLoader,
    _validate_points,
    _validate_text,
    _yaml_error_detail,
    quality_config_digest,
)

DEFAULT_PLUGIN_QUALITY_CONFIG = files("skillevaluator.config").joinpath("plugin-quality-default.yaml")

DIMENSIONS = ("correctness", "discoverability", "reliability", "efficiency")
SEVERITIES = ("error", "warning", "info")
APPLIES_TO = ("all", "claude")
_BASE_KEYS = frozenset({"dimension", "severity", "points", "applies_to", "message", "suggestion", "source"})
_FIXED_KEYS = ("dimension", "applies_to")
_INT_PARAMS = frozenset({"min_chars", "max_chars", "max_tokens", "cap"})
_LIST_PARAMS = frozenset({"terms", "events", "headings", "patterns", "interpreters"})
_PARAM_KEYS = _INT_PARAMS | _LIST_PARAMS | {"markers"}


class PluginQualityConfigError(QualityConfigError):
    """Raised when a plugin quality configuration is invalid."""


class PluginQualityRule(TypedDict, total=False):
    """One plugin quality rule: where it deducts, how much, and its parameters."""

    dimension: str
    severity: Literal["error", "warning", "info"]
    points: int | float
    applies_to: Literal["all", "claude"]
    message: str
    suggestion: str
    source: str
    min_chars: int
    max_chars: int
    max_tokens: int
    cap: int
    terms: list[str]
    events: list[str]
    headings: list[str]
    patterns: list[str]
    interpreters: list[str]
    markers: dict[str, list[str]]


def load_plugin_quality_config(path: Path | None = None) -> dict[str, PluginQualityRule]:
    """Load and validate the packaged rules, or a custom file merged over them."""
    return _load(path)[0]


def load_plugin_quality_config_with_provenance(
    path: Path | None = None,
) -> tuple[dict[str, PluginQualityRule], QualityConfigProvenance]:
    """Load the rules and describe them. Only the file name is kept, so reports do not leak local paths.

    The digest covers the rules the file itself declares, as for the skill quality config.
    """
    rules, declared = _load(path)
    provenance: QualityConfigProvenance = {
        "source": "custom" if path else "packaged",
        "file": path.name if path else None,
        "digest": quality_config_digest(declared),
    }
    return rules, provenance


def _read_yaml(path: Path | None, source: str) -> object:
    try:
        text = path.read_text(encoding="utf-8") if path else DEFAULT_PLUGIN_QUALITY_CONFIG.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PluginQualityConfigError(f"{source}: file is not valid UTF-8") from exc
    except OSError as exc:
        raise PluginQualityConfigError(f"{source}: cannot read file: {exc.strerror or exc}") from exc
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)  # nosec B506
    except yaml.YAMLError as exc:
        raise PluginQualityConfigError(f"{source}: invalid YAML: {_yaml_error_detail(exc)}") from exc
    except RecursionError as exc:
        raise PluginQualityConfigError(f"{source}: invalid YAML: nesting is too deep") from exc
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PluginQualityConfigError(f"{source}: invalid YAML: cannot parse a value: {exc}") from exc


def _rules_mapping(raw: object, source: str) -> dict:
    if not isinstance(raw, dict):
        raise PluginQualityConfigError(f"{source}: expected a mapping with a top-level 'rules' key")
    try:
        _reject_unknown_keys(raw, frozenset({"rules"}), source, "top level")
    except QualityConfigError as exc:
        raise PluginQualityConfigError(str(exc)) from exc
    rules = raw.get("rules")
    if not isinstance(rules, dict):
        raise PluginQualityConfigError(f"{source}: rules must be a mapping of rule IDs")
    return rules


def _validate_list(value: object, source: str, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PluginQualityConfigError(f"{source}: {location} must be a non-empty list")
    for index, entry in enumerate(value):
        if not isinstance(entry, str) or not entry.strip():
            raise PluginQualityConfigError(f"{source}: {location}[{index}] must be a non-blank string")
    return list(value)


def _validate_rule(rule: dict, source: str, location: str) -> PluginQualityRule:
    """Check one fully merged rule and return a normalized copy."""
    try:
        points = _validate_points(rule.get("points"), source, location)
        for key in ("message", "suggestion", "source"):
            _validate_text(rule.get(key), source, f"{location}.{key}")
    except QualityConfigError as exc:
        raise PluginQualityConfigError(str(exc)) from exc
    if rule.get("dimension") not in DIMENSIONS:
        raise PluginQualityConfigError(f"{source}: {location}.dimension must be one of {', '.join(DIMENSIONS)}")
    if rule.get("severity") not in SEVERITIES:
        raise PluginQualityConfigError(f"{source}: {location}.severity must be one of {', '.join(SEVERITIES)}")
    if rule.get("applies_to") not in APPLIES_TO:
        raise PluginQualityConfigError(f"{source}: {location}.applies_to must be one of {', '.join(APPLIES_TO)}")
    normalized: dict = {key: rule[key] for key in _BASE_KEYS}
    normalized["points"] = points
    for key in sorted(rule.keys() & _PARAM_KEYS):
        value = rule[key]
        if key in _INT_PARAMS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PluginQualityConfigError(f"{source}: {location}.{key} must be a non-negative integer")
            normalized[key] = value
        elif key in _LIST_PARAMS:
            normalized[key] = _validate_list(value, source, f"{location}.{key}")
        else:  # markers: language -> list of substrings
            if not isinstance(value, dict) or not value:
                raise PluginQualityConfigError(f"{source}: {location}.markers must be a non-empty mapping")
            normalized[key] = {
                str(lang): _validate_list(entries, source, f"{location}.markers.{lang}")
                for lang, entries in value.items()
            }
    return normalized  # type: ignore[return-value]


def _load(path: Path | None) -> tuple[dict[str, PluginQualityRule], dict]:
    packaged_source = "packaged plugin-quality-default.yaml"
    packaged_raw = _rules_mapping(_read_yaml(None, packaged_source), packaged_source)
    packaged: dict[str, PluginQualityRule] = {}
    for rule_id, rule in packaged_raw.items():
        location = f"rules.{rule_id}"
        if not isinstance(rule, dict):
            raise PluginQualityConfigError(f"{packaged_source}: {location} must be a mapping")
        missing = sorted(_BASE_KEYS - rule.keys())
        if missing:
            raise PluginQualityConfigError(f"{packaged_source}: {location} is missing key(s): {', '.join(missing)}")
        try:
            _reject_unknown_keys(rule, _BASE_KEYS | _PARAM_KEYS, packaged_source, location)
        except QualityConfigError as exc:
            raise PluginQualityConfigError(str(exc)) from exc
        packaged[str(rule_id)] = _validate_rule(rule, packaged_source, location)
    if path is None:
        return packaged, {"rules": packaged}

    source = str(path)
    custom_raw = _rules_mapping(_read_yaml(path, source), source)
    merged = {rule_id: dict(rule) for rule_id, rule in packaged.items()}
    declared: dict[str, dict] = {}
    for rule_id, block in custom_raw.items():
        location = f"rules.{rule_id}"
        if rule_id not in packaged:
            raise PluginQualityConfigError(
                f"{source}: unknown rule '{rule_id}' (known rules: {', '.join(sorted(packaged))})"
            )
        if block is None:
            block = {}
        if not isinstance(block, dict):
            raise PluginQualityConfigError(f"{source}: {location} must be a mapping")
        for key in _FIXED_KEYS:
            if key in block:
                raise PluginQualityConfigError(f"{source}: {location}.{key} cannot be changed in a custom file")
        allowed = (_BASE_KEYS - set(_FIXED_KEYS)) | (packaged[rule_id].keys() & _PARAM_KEYS)
        try:
            _reject_unknown_keys(block, frozenset(allowed), source, location)
        except QualityConfigError as exc:
            raise PluginQualityConfigError(str(exc)) from exc
        rule = _validate_rule({**packaged[rule_id], **block}, source, location)
        merged[rule_id] = rule
        if block:
            declared[rule_id] = {key: rule[key] for key in block}  # type: ignore[literal-required]
    return merged, {"rules": declared}  # type: ignore[return-value]
