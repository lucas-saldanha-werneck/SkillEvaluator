# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin quality score: regressions found in review (false positives, a crash, identity, modes)."""

from __future__ import annotations

import json

import pytest
from tests.validators.plugin_quality_support import GOOD_MANIFEST, make_plugin

from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.validators.plugin_quality_score import PluginQualityScoreValidator

README = "# x\n\n## Limitations\nnone\n"
COMMON = {"README.md": README, "tests/test_a.py": ""}


def _score(root, **kw):
    return PluginQualityScoreValidator(**kw).score(plugin_inventory_for_root(root))


def _ids(qs):
    return [i.rule_id for i in qs.all_issues]


def _hk(command, event="PreToolUse", matcher="Bash", **extra):
    handler = {"type": "command", "command": command, "timeout": 5, "statusMessage": "x", **extra}
    group = {"hooks": [handler]}
    if matcher is not None:
        group["matcher"] = matcher
    return json.dumps({"hooks": {event: [group]}})


def _plugin(tmp_path, files, **kw):
    return make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={**COMMON, **files}, **kw)


# --- P1: hooks declared in frontmatter ---------------------------------------------


@pytest.mark.parametrize("path", ["commands/fmt.md", "skills/fmt/SKILL.md"])
def test_frontmatter_hooks_are_not_an_unwrapped_hooks_file(tmp_path, path):
    body = (
        "---\nname: fmt\ndescription: Formats the tables in the current pull request body\nhooks:\n"
        "  PreToolUse:\n    - matcher: Bash\n      hooks:\n        - type: command\n          command: echo hi\n"
        "---\nbody\n"
    )
    qs = _score(_plugin(tmp_path, {path: body}))
    ids = _ids(qs)
    assert "hooks_file_unwrapped" not in ids
    assert "hook_timeout_missing" in ids  # the other hook rules still read them


# --- P1: validate never raises on odd command paths ----------------------------------


@pytest.mark.parametrize(
    "command", ["bash scripts//run.sh", "bash ./scripts/./run.sh", "bash scripts/../scripts/run.sh"]
)
def test_unnormalized_script_path_does_not_raise(tmp_path, command):
    root = _plugin(tmp_path, {"hooks/hooks.json": _hk(command), "scripts/run.sh": "set -e\n"})
    result = PluginQualityScoreValidator().validate(root)
    assert result.metadata["plugin_quality_scores"]["overall_score"] > 0
    qs = _score(root)
    issue = next(i for i in qs.all_issues if i.rule_id == "hook_without_plugin_root")
    assert issue.file == "hooks/hooks.json"


# --- P2: OpenClaw identity and author ------------------------------------------------


def _openclaw(tmp_path, manifest):
    root = tmp_path / "oc"
    root.mkdir()
    (root / "openclaw.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "README.md").write_text(README, encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("", encoding="utf-8")
    return root


OPENCLAW = {
    "id": "table-tools",
    "name": "Table Tools Display Label",
    "configSchema": {"type": "object"},
    "version": "1.0.0",
    "description": "Formats Markdown tables in pull request descriptions and checks their column widths.",
}


def test_openclaw_is_scored_by_its_id_not_its_display_name(tmp_path):
    qs = _score(_openclaw(tmp_path, OPENCLAW))
    assert qs.plugin_name == "table-tools"
    assert "name_format" not in _ids(qs)


def test_openclaw_id_is_checked_for_reserved_terms(tmp_path):
    qs = _score(_openclaw(tmp_path, {**OPENCLAW, "id": "claude-tables", "name": "Fine Name"}))
    assert "name_reserved" in _ids(qs)


def test_author_is_not_applicable_to_openclaw(tmp_path):
    qs = _score(_openclaw(tmp_path, OPENCLAW))
    assert "author_missing" not in _ids(qs)
    assert "author_missing" in qs.not_applicable


def test_author_missing_still_applies_to_claude(tmp_path):
    manifest = {k: v for k, v in GOOD_MANIFEST.items() if k != "author"}
    qs = _score(make_plugin(tmp_path / "p", manifest=manifest, files=COMMON))
    assert "author_missing" in _ids(qs)
    assert "author_missing" not in qs.not_applicable


# --- P2: state written under the plugin root ----------------------------------------


def _state_ids(tmp_path, script):
    root = _plugin(tmp_path, {"hooks/hooks.json": _hk('python3 "${CLAUDE_PLUGIN_ROOT}/s.py"'), "s.py": script})
    return _ids(_score(root))


@pytest.mark.parametrize(
    "script",
    [
        "import os\ntry:\n  pass\nexcept Exception: pass\n"
        "print(open(os.path.join(os.environ['CLAUDE_PLUGIN_ROOT'], 'data.json')).read())\n",
        "import os\ntry:\n  pass\nexcept Exception: pass\n"
        "print(open(os.path.join(os.environ['CLAUDE_PLUGIN_ROOT'], 'data.json'), 'r').read())\n",
        "import os\ntry:\n  pass\nexcept Exception: pass\n"
        "open(os.environ['CLAUDE_PLUGIN_DATA'] + '/x', 'w').write(os.environ['CLAUDE_PLUGIN_ROOT'])\n",
        "const fs = require('fs')\ntry {} catch (e) {}\n"
        "fs.writeFileSync(`${process.env.CLAUDE_PLUGIN_DATA}/x`, process.env.CLAUDE_PLUGIN_ROOT)\n",
    ],
)
def test_reads_and_data_writes_are_not_state_in_plugin_root(tmp_path, script):
    assert "state_in_plugin_root" not in _state_ids(tmp_path, script)


@pytest.mark.parametrize(
    "script",
    [
        "import os\ntry:\n  pass\nexcept Exception: pass\nopen(os.environ['CLAUDE_PLUGIN_ROOT'] + '/x.json', 'w').write('{}')\n",
        "import os\ntry:\n  pass\nexcept Exception: pass\n"
        "open(os.path.join(os.environ['CLAUDE_PLUGIN_ROOT'], 'x.json'), mode='a').write('{}')\n",
        "import os, pathlib\ntry:\n  pass\nexcept Exception: pass\n"
        "(pathlib.Path(os.environ['CLAUDE_PLUGIN_ROOT']) / 'x').write_text('{}')\n",
    ],
)
def test_real_writes_under_the_root_still_count(tmp_path, script):
    assert "state_in_plugin_root" in _state_ids(tmp_path, script)


# --- P3 ------------------------------------------------------------------------------


def test_unknown_handler_type_is_unknown(tmp_path):
    hooks = {"hooks": {"Stop": [{"hooks": [{"type": "foo", "command": "x"}]}]}}
    assert "hook_event_unknown" in _ids(_score(_plugin(tmp_path, {"hooks/hooks.json": json.dumps(hooks)})))


@pytest.mark.parametrize("handler", [{"type": "prompt", "prompt": "p"}, {"type": "http", "url": "https://e.example"}])
def test_documented_handler_types_are_known(tmp_path, handler):
    hooks = {"hooks": {"Stop": [{"hooks": [handler]}]}}
    assert "hook_event_unknown" not in _ids(_score(_plugin(tmp_path, {"hooks/hooks.json": json.dumps(hooks)})))


@pytest.mark.parametrize(
    "command", ['cd "${CLAUDE_PLUGIN_ROOT}" && ./scripts/x.sh', "cd $CLAUDE_PLUGIN_ROOT/scripts && ./x.sh"]
)
def test_cd_to_plugin_root_then_relative_script_is_rooted(tmp_path, command):
    root = _plugin(
        tmp_path, {"hooks/hooks.json": _hk(command), "scripts/x.sh": "set -e\n"}, executable=["scripts/x.sh"]
    )
    assert "hook_without_plugin_root" not in _ids(_score(root))


def test_relative_script_without_cd_still_flagged(tmp_path):
    root = _plugin(tmp_path, {"hooks/hooks.json": _hk("./scripts/x.sh"), "scripts/x.sh": "set -e\n"})
    assert "hook_without_plugin_root" in _ids(_score(root))


@pytest.mark.parametrize("matcher", ["*", "", None, ".*"])
def test_match_all_matchers_are_broad(tmp_path, matcher):
    assert "hook_matcher_broad" in _ids(_score(_plugin(tmp_path, {"hooks/hooks.json": _hk("true", matcher=matcher)})))


def test_mod_filter_in_a_variable_is_filtered(tmp_path):
    files = {
        "hooks/hooks.json": json.dumps({"modules": ["./register.ts"]}),
        "hooks/register.ts": "const f = { tool: 'Bash' }\nexport function register(on) {\n  on('tool.call', f, (e) => e)\n}\n",
    }
    assert "mod_unfiltered_handler" not in _ids(_score(_plugin(tmp_path, files)))


def test_mod_handler_passed_by_name_is_still_unfiltered(tmp_path):
    files = {
        "hooks/hooks.json": json.dumps({"modules": ["./register.ts"]}),
        "hooks/register.ts": "export function register(on) {\n  on('tool.call', handler)\n}\n",
    }
    assert "mod_unfiltered_handler" in _ids(_score(_plugin(tmp_path, files)))


def test_no_manifest_is_medium_in_informational_mode_and_high_when_enforced(tmp_path):
    (tmp_path / "empty").mkdir()
    advisory = PluginQualityScoreValidator().validate(tmp_path / "empty")
    assert advisory.passed is True
    assert advisory.findings[0].check_name == "plugin_manifest_missing"
    assert advisory.findings[0].severity.value.lower() == "medium"
    enforced = PluginQualityScoreValidator(enforce=True).validate(tmp_path / "empty")
    assert enforced.passed is False
    assert enforced.findings[0].severity.value.lower() == "high"


def test_claude_only_rules_are_read_lazily():
    from skillevaluator.validators import plugin_quality_score as module

    assert not hasattr(module, "CLAUDE_ONLY_RULES")
    assert "hook_timeout_missing" in module.claude_only_rules()


# --- P3: a command without a description uses its first line -------------------------


def test_command_without_description_uses_its_first_body_line(tmp_path):
    files = {"commands/fmt.md": "Format the Markdown tables in the current pull request body.\n\nMore.\n"}
    assert "component_description_missing" not in _ids(_score(_plugin(tmp_path, files)))


def test_command_with_frontmatter_but_no_description_uses_first_body_line(tmp_path):
    files = {"commands/fmt.md": "---\nmodel: haiku\n---\n\n# Format the Markdown tables in the pull request\n"}
    assert "component_description_missing" not in _ids(_score(_plugin(tmp_path, files)))


@pytest.mark.parametrize("body", ["", "---\nmodel: haiku\n---\n", "```\ncode\n```\n", "go\n"])
def test_command_without_a_useful_first_line_is_still_flagged(tmp_path, body):
    assert "component_description_missing" in _ids(_score(_plugin(tmp_path, {"commands/fmt.md": body})))


def test_agent_without_description_is_still_flagged(tmp_path):
    files = {"agents/fmt.md": "Formats the Markdown tables in the current pull request body.\n"}
    assert "component_description_missing" in _ids(_score(_plugin(tmp_path, files)))
