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


# --- Discoverability -------------------------------------------------------------


@pytest.mark.parametrize(
    "desc,rule,points",
    [
        ("Short", "description_short", 20),
        ("x" * 201, "description_long", 5),
        ("Handles various things for your project setup and other tasks", "description_vague", 15),
        ("You can use this to format Markdown tables in pull requests", "description_person", 15),
    ],
)
def test_plugin_description_rules(tmp_path, desc, rule, points):
    qs = _score(
        make_plugin(tmp_path / "p", manifest={**GOOD_MANIFEST, "description": desc}, files={"README.md": "# p"})
    )
    assert {i.rule_id: i.deduction for i in qs.discoverability.issues}[rule] == points


def test_good_description_has_no_description_issue(tmp_path):
    qs = _score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"README.md": "# p"}))
    assert not [i for i in _ids(qs) if i.startswith("description_")]


def test_missing_description_counts_as_short(tmp_path):
    m = {k: v for k, v in GOOD_MANIFEST.items() if k != "description"}
    qs = _score(make_plugin(tmp_path / "p", manifest=m))
    issue = next(i for i in qs.all_issues if i.rule_id == "description_short")
    assert "missing" in issue.message


def test_component_description_cap(tmp_path):
    files = {f"commands/c{i}.md": "Do it\n" for i in range(4)}
    qs = _score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={**files, "README.md": "# p"}))
    [issue] = [i for i in qs.all_issues if i.rule_id == "component_description_missing"]
    assert issue.deduction == 30


def test_described_command_and_agent_are_fine(tmp_path):
    files = {
        "commands/c.md": "---\ndescription: Formats the current table\n---\nDo it\n",
        "agents/a.md": "---\nname: a\ndescription: Reviews table widths\n---\nBody\n",
        "README.md": "# p",
    }
    qs = _score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files=files))
    assert "component_description_missing" not in _ids(qs)


def test_mod_registration_short_description(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "README.md": "# p",
                "hooks/hooks.json": json.dumps({"modules": ["./r.ts"]}),
                "hooks/r.ts": "export function register(on) { on('session.start', async ($) => {"
                " await $.tool.register({ name: 'a', description: 'Too short' })"
                " await $.tool.register({ name: 'b', description: D }) }) }\n",
            },
        )
    )
    assert {i.rule_id: i.deduction for i in qs.all_issues}["mod_registration_description"] == 10
    assert qs.unresolved_registrations == 1


def test_readme_missing_and_status_message(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "hooks/hooks.json": json.dumps(
                    {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true", "timeout": 5}]}]}}
                )
            },
        )
    )
    assert {"readme_missing", "hook_status_message_missing"} <= set(_ids(qs))


def test_status_message_satisfies_transparency(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "readme.md": "# p",
                "hooks/hooks.json": json.dumps(
                    {
                        "hooks": {
                            "Stop": [
                                {"hooks": [{"type": "command", "command": "true", "timeout": 5, "statusMessage": "Saving"}]}
                            ]
                        }
                    }
                ),
            },
        )
    )
    assert not {"readme_missing", "hook_status_message_missing"} & set(_ids(qs))


# --- Reliability -----------------------------------------------------------------


def test_mod_without_tests_counts_once(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "README.md": "# p",
                "hooks/hooks.json": json.dumps({"modules": ["./r.ts"]}),
                "hooks/r.ts": "export function register(on) {}\n",
            },
        )
    )
    ids = _ids(qs)
    assert "mod_tests_missing" in ids and "tests_missing" not in ids


def test_mod_with_tests_is_fine(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "hooks/hooks.json": json.dumps({"modules": ["./r.ts"]}),
                "hooks/r.ts": "export function register(on) {}\n",
                "tests/r.test.ts": "test('x', () => {})\n",
            },
        )
    )
    assert not {"mod_tests_missing", "tests_missing"} & set(_ids(qs))


def test_plain_plugin_without_tests(tmp_path):
    assert "tests_missing" in _ids(_score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"README.md": "# p"})))


@pytest.mark.parametrize("test_file", ["tests/test_x.py", "scripts/run_test.go", "lib/a.test.js", "test/x.bats"])
def test_test_files_satisfy_tests_rule(tmp_path, test_file):
    ids = _ids(
        _score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"README.md": "# p", test_file: "x\n"}))
    )
    assert "tests_missing" not in ids


def test_hook_script_rules(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "README.md": "# p\n## Limitations\n- none\n",
                "hooks/hooks.json": json.dumps(
                    {
                        "description": "d",
                        "hooks": {
                            "Stop": [
                                {
                                    "hooks": [
                                        {"type": "command", "command": '"${CLAUDE_PLUGIN_ROOT}/run.sh"'},
                                        {
                                            "type": "command",
                                            "command": 'python3 "${CLAUDE_PLUGIN_ROOT}/ok.py"',
                                            "timeout": 5,
                                        },
                                    ]
                                }
                            ]
                        },
                    }
                ),
                "run.sh": "echo hi\n",
                "ok.py": "try:\n    pass\nexcept Exception:\n    raise\n",
            },
        )
    )
    by_id = {i.rule_id: i for i in qs.all_issues}
    assert "run.sh" in by_id["hook_script_error_handling"].message
    assert "ok.py" not in by_id["hook_script_error_handling"].message
    assert by_id["hook_timeout_missing"].deduction == 5
    assert "run.sh" in by_id["script_not_executable"].message
    assert "ok.py" not in by_id["script_not_executable"].message
    assert "readme_limitations_missing" not in by_id


def test_executable_script_is_fine(tmp_path):
    ids = _ids(
        _score(
            make_plugin(
                tmp_path / "p",
                manifest=GOOD_MANIFEST,
                executable=["run.sh"],
                files={
                    "hooks/hooks.json": json.dumps(
                        {
                            "description": "d",
                            "hooks": {
                                "Stop": [
                                    {
                                        "hooks": [
                                            {"type": "command", "command": '"${CLAUDE_PLUGIN_ROOT}/run.sh"', "timeout": 5}
                                        ]
                                    }
                                ]
                            },
                        }
                    ),
                    "run.sh": "set -e\necho hi\n",
                },
            )
        )
    )
    assert "script_not_executable" not in ids and "hook_script_error_handling" not in ids


def test_file_read_by_a_command_is_not_a_direct_script(tmp_path):
    ids = _ids(
        _score(
            make_plugin(
                tmp_path / "p",
                manifest=GOOD_MANIFEST,
                files={
                    "hooks/hooks.json": json.dumps(
                        {
                            "description": "d",
                            "hooks": {
                                "SessionStart": [
                                    {
                                        "hooks": [
                                            {"type": "command", "command": 'cat "${CLAUDE_PLUGIN_ROOT}/ctx.sh"', "timeout": 5}
                                        ]
                                    }
                                ]
                            },
                        }
                    ),
                    "ctx.sh": "echo hi\n",
                },
            )
        )
    )
    assert "script_not_executable" not in ids


def test_readme_without_limitations(tmp_path):
    ids = _ids(_score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"README.md": "# p\n## Usage\nx\n"})))
    assert "readme_limitations_missing" in ids


# --- Efficiency ------------------------------------------------------------------


def _hooks_file(event, matcher, command="true"):
    group = {"hooks": [{"type": "command", "command": command, "timeout": 5}]}
    if matcher is not None:
        group["matcher"] = matcher
    return {"hooks/hooks.json": json.dumps({"description": "d", "hooks": {event: [group]}})}


@pytest.mark.parametrize("matcher", ["*", "", None])
def test_broad_pretooluse_matcher(tmp_path, matcher):
    ids = _ids(_score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files=_hooks_file("PreToolUse", matcher))))
    assert "hook_matcher_broad" in ids


def test_named_matcher_is_fine(tmp_path):
    ids = _ids(_score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files=_hooks_file("PreToolUse", "Bash"))))
    assert "hook_matcher_broad" not in ids


def test_broad_matcher_on_other_event_is_fine(tmp_path):
    ids = _ids(_score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files=_hooks_file("Stop", None))))
    assert "hook_matcher_broad" not in ids


def test_large_always_on_context(tmp_path):
    files = {f"commands/c{i}.md": f"---\ndescription: {'w' * 1000}\n---\nbody\n" for i in range(25)}
    qs = _score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files=files))
    assert qs.always_on_tokens > 5000 and "context_cost_high" in _ids(qs)


def test_small_context_is_fine(tmp_path):
    qs = _score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"README.md": "# p"}))
    assert "context_cost_high" not in _ids(qs)


def test_large_session_start_text(tmp_path):
    big = "x" * 3000
    files = _hooks_file("SessionStart", None, f"echo {big}")
    ids = _ids(_score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files=files)))
    assert "hook_static_context_large" in ids


def test_small_session_start_text_is_fine(tmp_path):
    files = _hooks_file("SessionStart", None, "echo hello")
    ids = _ids(_score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files=files)))
    assert "hook_static_context_large" not in ids


def test_unfiltered_tool_call_handler(tmp_path):
    ids = _ids(
        _score(
            make_plugin(
                tmp_path / "p",
                manifest=GOOD_MANIFEST,
                files={
                    "hooks/hooks.json": json.dumps({"modules": ["./r.ts"]}),
                    "hooks/r.ts": "export function register(on) { on('tool.call', async ($, e, next) => next(e)) }\n",
                },
            )
        )
    )
    assert "mod_unfiltered_handler" in ids


def test_filtered_handler_and_other_events_are_fine(tmp_path):
    ids = _ids(
        _score(
            make_plugin(
                tmp_path / "p",
                manifest=GOOD_MANIFEST,
                files={
                    "hooks/hooks.json": json.dumps({"modules": ["./r.ts"]}),
                    "hooks/r.ts": "export function register(on) {\n"
                    " on('ui.render', { component: 'Spinner' }, async ($, e, next) => next(e))\n"
                    " on('session.start', async ($, e, next) => next(e)) }\n",
                },
            )
        )
    )
    assert "mod_unfiltered_handler" not in ids


def test_large_file_cat_on_prompt_submit(tmp_path):
    files = _hooks_file("UserPromptSubmit", None, 'cat "${CLAUDE_PLUGIN_ROOT}/ctx.md"')
    files["ctx.md"] = "rule\n" * 600
    ids = _ids(_score(make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files=files)))
    assert "hook_static_context_large" in ids


# --- Aggregation, bundled skills, advisory and enforced modes ----------------------

GOOD_SKILL = (
    "---\nname: good-skill\ndescription: Formats CSV files into tables. Use when a CSV needs review.\n"
    "version: 1.0.0\nmetadata:\n  author: A\n  tags: [csv]\n---\n# Good\n## Purpose\nx\n## Examples\n```\nx\n```\n"
    "## Prerequisites\nx\n## Limitations\nx\n## Troubleshooting\nerror handling\n"
)


def _validate(root, **kw):
    return PluginQualityScoreValidator(**kw).validate(root)


def test_advisory_mode_never_fails(tmp_path):
    root = make_plugin(tmp_path / "p", manifest={"name": "Bad Name"}, files={"hooks/hooks.json": "{}"})
    result = _validate(root)
    pq = result.metadata["plugin_quality_scores"]
    assert result.passed is True and pq["passed"] is False and pq["enforced"] is False
    assert any(f.metadata.get("would_block") for f in result.findings)


def test_enforced_mode_fails_on_error_rule(tmp_path):
    root = make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"hooks/hooks.json": "{}"})
    result = _validate(root, enforce=True)
    assert result.passed is False
    assert "error rule: hooks_file_unwrapped" in result.metadata["plugin_quality_scores"]["blocking"]


def test_enforced_mode_fails_below_min_score(tmp_path):
    root = make_plugin(tmp_path / "p", manifest={"name": "x"})
    result = _validate(root, enforce=True, min_score=95)
    blocking = result.metadata["plugin_quality_scores"]["blocking"]
    assert result.passed is False and any(b.startswith("score ") and b.endswith("< 95") for b in blocking)


def test_failing_bundled_skill_blocks_plugin(tmp_path):
    root = make_plugin(
        tmp_path / "p",
        manifest=GOOD_MANIFEST,
        files={
            "README.md": "# p\n## Limitations\nx\n",
            "tests/test_a.py": "def test_a(): pass\n",
            "skills/good-skill/SKILL.md": GOOD_SKILL,
            "skills/bad/SKILL.md": "---\nname: Bad Skill\ndescription: x\n---\nhi\n",  # invalid name: error
        },
    )
    pq = _validate(root, enforce=True).metadata["plugin_quality_scores"]
    assert pq["skill_count"] == 2
    assert {s["skill_name"]: s["passed"] for s in pq["skills"]} == {"good-skill": True, "bad": False}
    assert pq["passed"] is False and "bundled skill 'bad' failed quality" in pq["blocking"]


def test_skills_only_plugin_passes_on_its_parts(tmp_path):
    root = make_plugin(
        tmp_path / "p",
        manifest=GOOD_MANIFEST,
        files={
            "README.md": "# p\n## Limitations\nx\n",
            "tests/test_a.py": "def test_a(): pass\n",
            "skills/good-skill/SKILL.md": GOOD_SKILL,
        },
    )
    result = _validate(root, enforce=True)
    assert result.passed is True and result.metadata["plugin_quality_scores"]["grade"] == "A"
    assert result.metadata["plugin_quality_config"]["source"] == "packaged"


def test_no_manifest_is_high(tmp_path):
    (tmp_path / "empty").mkdir()
    result = _validate(tmp_path / "empty", enforce=True)
    assert result.passed is False and result.findings[0].check_name == "plugin_manifest_missing"


# --- Exec form (command + args) ----------------------------------------------------


def test_exec_form_args_are_read_like_the_command(tmp_path):
    qs = _score(
        make_plugin(
            tmp_path / "p",
            manifest=GOOD_MANIFEST,
            files={
                "README.md": "# p",
                "hooks/hooks.json": json.dumps(
                    {
                        "description": "d",
                        "hooks": {
                            "Stop": [
                                {
                                    "hooks": [
                                        {"type": "command", "command": "bash", "args": ["scripts/a.sh"], "timeout": 5},
                                        {
                                            "type": "command",
                                            "command": "python3",
                                            "args": ["${CLAUDE_PLUGIN_ROOT}/b.py"],
                                            "timeout": 5,
                                        },
                                    ]
                                }
                            ]
                        },
                    }
                ),
                "scripts/a.sh": "set -e\necho ok\n",
                "b.py": "print('no handling')\n",
            },
        )
    )
    by_id = {i.rule_id: i for i in qs.all_issues}
    assert "scripts/a.sh" in by_id["hook_without_plugin_root"].message
    assert "b.py" in by_id["hook_script_error_handling"].message
    assert "script_not_executable" not in by_id


def test_exec_form_direct_script_needs_execute_bit(tmp_path):
    ids = _ids(
        _score(
            make_plugin(
                tmp_path / "p",
                manifest=GOOD_MANIFEST,
                files={
                    "hooks/hooks.json": json.dumps(
                        {
                            "description": "d",
                            "hooks": {
                                "Stop": [
                                    {
                                        "hooks": [
                                            {
                                                "type": "command",
                                                "command": "${CLAUDE_PLUGIN_ROOT}/run.sh",
                                                "args": ["x"],
                                                "timeout": 5,
                                            }
                                        ]
                                    }
                                ]
                            },
                        }
                    ),
                    "run.sh": "set -e\n",
                },
            )
        )
    )
    assert "script_not_executable" in ids
