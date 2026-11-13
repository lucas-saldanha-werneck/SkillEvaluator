# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON and console output of the plugin quality score."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console
from tests.validators.plugin_quality_support import GOOD_MANIFEST, make_plugin

from skillevaluator.reporting import CLIReporter, JSONReporter
from skillevaluator.tier1.commands import run_plugin_quality_check


def render_cli(results) -> str:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, record=True, width=200)
    for result in results:
        CLIReporter().render_result(result, console)
    return console.export_text()


def test_json_report_has_plugin_quality_block(tmp_path):
    root = make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST)
    results = run_plugin_quality_check(root)
    data = json.loads(JSONReporter().render_all(results))
    pq = data["plugin_quality"]
    assert pq["plugin_name"] == GOOD_MANIFEST["name"]
    assert set(pq["dimensions"]) == {"correctness", "discoverability", "reliability", "efficiency"}
    assert pq["config"]["source"] == "packaged" and "quality_summary" not in data


def test_console_shows_plugin_grade_and_mode(tmp_path):
    root = make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"hooks/hooks.json": "{}"})
    out = render_cli(run_plugin_quality_check(root, enforce=False))
    assert "Plugin:" in out and "advisory" in out and "Would block" in out and "hooks_file_unwrapped" in out
    assert "Plugin Quality Scores" in out


def test_console_lists_bundled_skills_and_not_applicable(tmp_path):
    root = tmp_path / "c"
    (root / ".cursor-plugin").mkdir(parents=True)
    (root / ".cursor-plugin" / "plugin.json").write_text(json.dumps({"name": "cur-demo"}), encoding="utf-8")
    (root / "skills" / "s").mkdir(parents=True)
    (root / "skills" / "s" / "SKILL.md").write_text("---\nname: s\ndescription: x\n---\nhi\n", encoding="utf-8")
    out = render_cli(run_plugin_quality_check(root))
    assert "enforced" in out and "Not applicable:" in out and "hook_timeout_missing" in out
