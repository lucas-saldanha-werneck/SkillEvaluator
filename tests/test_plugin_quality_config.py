# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the packaged plugin quality rules and custom overrides."""

from __future__ import annotations

import pytest

from skillevaluator.plugin_quality_config import (
    PluginQualityConfigError,
    load_plugin_quality_config,
    load_plugin_quality_config_with_provenance,
)

EXPECTED_IDS = {  # exactly the rule catalogue
    "manifest_invalid", "hooks_file_unwrapped", "component_path_missing", "component_path_outside_root",
    "name_format", "name_reserved", "author_missing", "version_missing", "hook_without_plugin_root",
    "state_in_plugin_root", "hook_event_unknown", "mod_module_invalid", "description_short",
    "description_long", "description_vague", "description_person", "component_description_missing",
    "mod_registration_description", "readme_missing", "hook_status_message_missing",
    "hook_script_error_handling", "hook_timeout_missing", "script_not_executable", "tests_missing",
    "mod_tests_missing", "readme_limitations_missing", "context_cost_high", "hook_matcher_broad",
    "hook_static_context_large", "mod_unfiltered_handler",
}  # fmt: skip


def test_packaged_config_lists_every_rule_with_source():
    rules = load_plugin_quality_config()
    assert set(rules) == EXPECTED_IDS
    assert all(r["source"] for r in rules.values())
    assert rules["description_short"]["points"] == 20 and rules["description_short"]["min_chars"] == 20
    assert rules["context_cost_high"]["max_tokens"] == 5000
    assert rules["hooks_file_unwrapped"]["severity"] == "error" and rules["hooks_file_unwrapped"]["applies_to"] == "claude"


def test_custom_file_overrides_points_and_keeps_rest(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("rules:\n  readme_missing:\n    points: 0\n", encoding="utf-8")
    rules, prov = load_plugin_quality_config_with_provenance(f)
    assert rules["readme_missing"]["points"] == 0
    assert rules["tests_missing"]["points"] == 10
    assert prov["source"] == "custom" and prov["file"] == "p.yaml" and prov["digest"].startswith("sha256:")


def test_packaged_provenance_has_no_file():
    _, prov = load_plugin_quality_config_with_provenance()
    assert prov["source"] == "packaged" and prov["file"] is None


def test_custom_list_replaces_packaged_list(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("rules:\n  hook_matcher_broad:\n    events: [PreToolUse]\n", encoding="utf-8")
    assert load_plugin_quality_config(f)["hook_matcher_broad"]["events"] == ["PreToolUse"]


@pytest.mark.parametrize(
    "body,needle",
    [
        ("rules:\n  nope:\n    points: 1\n", "unknown rule"),
        ("rules:\n  readme_missing:\n    dimension: efficiency\n", "dimension"),
        ("rules:\n  readme_missing:\n    points: -1\n", "points"),
        ("rules:\n  readme_missing:\n    severity: fatal\n", "severity"),
        ("rules:\n  readme_missing: {}\n  readme_missing: {}\n", "duplicate"),
        ("rules:\n  readme_missing:\n    applies_to: all\n", "applies_to"),
        ("rules:\n  readme_missing:\n    max_tokens: 3\n", "unknown key"),
        ("other: 1\n", "unknown key"),
        ("rules: []\n", "mapping"),
    ],
)
def test_invalid_custom_file_is_rejected(tmp_path, body, needle):
    f = tmp_path / "p.yaml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(PluginQualityConfigError, match=needle):
        load_plugin_quality_config(f)
