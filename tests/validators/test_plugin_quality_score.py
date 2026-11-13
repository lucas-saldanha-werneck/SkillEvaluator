# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin quality score: one fixture per rule (fires and does not fire)."""

from __future__ import annotations

import json

import pytest
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


# --- Correctness: hooks and mods -----------------------------------------------


def _hooks(event_map, extra=None):
    return json.dumps({"description": "Demo hooks", "hooks": event_map, **(extra or {})})


def test_unwrapped_hooks_file_is_error(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={"hooks/hooks.json": json.dumps({"PreToolUse": [{"matcher": "Bash", "hooks": []}]})},
        )
    )
    issue = next(i for i in qs.all_issues if i.rule_id == "hooks_file_unwrapped")
    assert issue.severity == "error" and issue.file == "hooks/hooks.json"


def test_hooks_and_modules_together_are_both_read(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "hooks/hooks.json": _hooks(
                    {"Stop": [{"hooks": [{"type": "command", "command": "true", "timeout": 5}]}]},
                    {"modules": ["./register.ts"]},
                ),
                "hooks/register.ts": "export function register(on) {}\n",
            },
        )
    )
    assert not {"hooks_file_unwrapped", "mod_module_invalid"} & set(_ids(qs))


@pytest.mark.xfail(reason="hook_timeout_missing is added with the reliability rules", strict=True)
def test_inline_manifest_hooks_are_not_unwrapped_file(tmp_path):
    manifest = {
        **GOOD_MANIFEST,
        "hooks": {"SessionStart": [{"matcher": "startup", "hooks": [{"type": "command", "command": "echo hi"}]}]},
    }
    ids = _ids(_score(make_plugin(tmp_path / "p", manifest=manifest)))
    assert "hooks_file_unwrapped" not in ids and "hook_timeout_missing" in ids


def test_unknown_event(tmp_path):
    ids = _ids(
        _score(
            make_plugin(
                tmp_path / "p",
                manifest=GOOD_MANIFEST,
                files={"hooks/hooks.json": _hooks({"OnBanana": [{"hooks": [{"type": "command", "command": "true"}]}]})},
            )
        )
    )
    assert "hook_event_unknown" in ids


def test_script_without_plugin_root_and_state_write(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "hooks/hooks.json": _hooks(
                    {
                        "Stop": [
                            {
                                "hooks": [
                                    {"type": "command", "command": "bash scripts/a.sh", "timeout": 5},
                                    {
                                        "type": "command",
                                        "command": 'bash "${CLAUDE_PLUGIN_ROOT}/scripts/b.sh"',
                                        "timeout": 5,
                                    },
                                ]
                            }
                        ]
                    }
                ),
                "scripts/a.sh": "set -e\necho ok\n",
                "scripts/b.sh": 'set -e\necho 1 >> "${CLAUDE_PLUGIN_ROOT}/state.log"\n',
            },
        )
    )
    by_id = {i.rule_id: i for i in qs.all_issues}
    assert "scripts/a.sh" in by_id["hook_without_plugin_root"].message
    assert "scripts/b.sh" not in by_id["hook_without_plugin_root"].message
    assert by_id["state_in_plugin_root"].file == "scripts/b.sh" and by_id["state_in_plugin_root"].line == 2


def test_python_state_write_in_plugin_root(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "hooks/hooks.json": _hooks(
                    {"Stop": [{"hooks": [{"type": "command", "command": 'python3 "${CLAUDE_PLUGIN_ROOT}/s.py"'}]}]}
                ),
                "s.py": "import os\nopen(os.environ['CLAUDE_PLUGIN_ROOT'] + '/x.json', 'w').write('{}')\n",
            },
        )
    )
    assert "state_in_plugin_root" in _ids(qs)


def test_plugin_data_write_is_fine(tmp_path):
    ids = _ids(
        _score(
            make_plugin(
                tmp_path / "p",
                manifest=GOOD_MANIFEST,
                files={
                    "hooks/hooks.json": _hooks(
                        {
                            "Stop": [
                                {
                                    "hooks": [
                                        {"type": "command", "command": 'bash "${CLAUDE_PLUGIN_ROOT}/s.sh"', "timeout": 5}
                                    ]
                                }
                            ]
                        }
                    ),
                    "s.sh": 'set -e\necho 1 >> "${CLAUDE_PLUGIN_DATA}/state.log"\n',
                },
            )
        )
    )
    assert "state_in_plugin_root" not in ids and "hook_without_plugin_root" not in ids


def test_mod_module_invalid_is_one_error(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p", manifest=GOOD_MANIFEST, files={"hooks/hooks.json": json.dumps({"modules": ["./gone.ts"]})}
        )
    )
    [issue] = [i for i in qs.all_issues if i.rule_id == "mod_module_invalid"]
    assert issue.severity == "error" and "./gone.ts" in issue.message
