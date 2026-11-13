# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin quality score: one fixture per rule (fires and does not fire)."""

from __future__ import annotations

import json

from tests.validators.plugin_quality_support import GOOD_MANIFEST, make_plugin

from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.validators.plugin_quality_score import PluginQualityScoreValidator


def _score(root, **kw):
    return PluginQualityScoreValidator(**kw).score(plugin_inventory_for_root(root))


def _ids(qs):
    return [i.rule_id for i in qs.all_issues]


# --- Correctness: manifest -----------------------------------------------------


def test_clean_manifest_has_no_correctness_issues(tmp_path):
    qs = _score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"README.md": "# p\n"}))
    assert qs.correctness.issues == []


def test_unparseable_manifest_scores_without_crash(tmp_path):
    qs = _score(make_plugin(tmp_path / "p", manifest="{not json"))
    assert _ids(qs).count("manifest_invalid") == 1


def test_bad_and_reserved_name_without_double_count(tmp_path):
    qs = _score(make_plugin(tmp_path / "p", manifest={**GOOD_MANIFEST, "name": "Claude_Tool"}))
    ids = _ids(qs)
    assert "name_reserved" in ids
    # the schema reports the name or name_format does, never both
    assert ids.count("manifest_invalid") + ids.count("name_format") == 1


def test_version_from_marketplace_entry_suppresses_version_missing(tmp_path):
    root = make_plugin(tmp_path / "p", manifest={k: v for k, v in GOOD_MANIFEST.items() if k != "version"})
    (root / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({"name": "m", "plugins": [{"name": GOOD_MANIFEST["name"], "source": "./", "version": "0.1.0"}]}),
        encoding="utf-8",
    )
    assert "version_missing" not in _ids(_score(root))


def test_missing_author_and_version_cost_five_each(tmp_path):
    qs = _score(make_plugin(tmp_path / "p", manifest={"name": "demo-plugin", "description": GOOD_MANIFEST["description"]}))
    assert {i.rule_id: i.deduction for i in qs.correctness.issues} == {"author_missing": 5, "version_missing": 5}


def test_declared_component_path_missing_and_escaping(tmp_path):
    qs = _score(make_plugin(tmp_path / "p", manifest={**GOOD_MANIFEST, "commands": ["./nope.md"], "agents": ["../x.md"]}))
    assert {"component_path_missing", "component_path_outside_root"} <= set(_ids(qs))


def test_non_claude_manifest_marks_claude_rules_not_applicable(tmp_path):
    root = tmp_path / "c"
    (root / ".cursor-plugin").mkdir(parents=True)
    (root / ".cursor-plugin" / "plugin.json").write_text(
        json.dumps({"name": "cur-demo", "version": "1.0.0"}), encoding="utf-8"
    )
    qs = _score(root)
    assert qs.manifest_type != "claude_plugin_json"
    assert {"hook_timeout_missing", "mod_module_invalid", "hook_without_plugin_root"} <= set(qs.not_applicable)
