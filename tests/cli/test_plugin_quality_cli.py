# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI wiring of the plugin quality score: the quality step for plugins, the command, and the flags."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from tests.validators.plugin_quality_support import GOOD_MANIFEST, make_plugin
from tests.validators.test_plugin_quality_score import GOOD_SKILL

from skillevaluator.cli import cli
from skillevaluator.tier1.commands import run_validation


@pytest.mark.xfail(reason="the JSON plugin_quality block is added with the reporters", strict=True)
def test_validate_plugin_reports_but_does_not_fail_by_default(tmp_path):
    root = make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"hooks/hooks.json": "{}"})
    r = CliRunner().invoke(
        cli,
        ["validate", str(root), "--type", "plugin", "--checks", "quality", "-r", "json", "-o", str(tmp_path / "out")],
    )
    report = json.loads(next((tmp_path / "out").glob("*.json")).read_text(encoding="utf-8"))
    assert report["plugin_quality"]["passed"] is False
    assert r.exit_code == 0, r.output


def test_validate_plugin_quality_step_runs_without_failing(tmp_path):
    root = make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"hooks/hooks.json": "{}"})
    results = run_validation(root, checks="quality", content_type="plugin")
    [pq] = [r for r in results if r.metadata.get("plugin_quality_scores")]
    assert pq.passed and pq.metadata["plugin_quality_scores"]["passed"] is False


def test_validate_plugin_enforced_fails(tmp_path):
    root = make_plugin(tmp_path / "p", manifest=GOOD_MANIFEST, files={"hooks/hooks.json": "{}"})
    r = CliRunner().invoke(
        cli,
        ["validate", str(root), "--type", "plugin", "--checks", "quality", "--enforce-plugin-quality", "-r", "cli"],
    )
    assert r.exit_code != 0


def test_plugin_quality_check_command_exit_codes(tmp_path):
    good = make_plugin(
        tmp_path / "g",
        manifest=GOOD_MANIFEST,
        files={"README.md": "# g\n## Limitations\nx\n", "tests/test_a.py": "def test_a(): pass\n"},
    )
    bad = make_plugin(tmp_path / "b", manifest=GOOD_MANIFEST, files={"hooks/hooks.json": "{}"})
    ok = CliRunner().invoke(cli, ["plugin-quality-check", str(good), "-r", "cli"])
    assert ok.exit_code == 0, ok.output
    assert CliRunner().invoke(cli, ["tier1", "plugin-quality-check", str(bad), "-r", "cli"]).exit_code != 0


def test_bad_plugin_quality_config_is_usage_error(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("rules:\n  nope: {}\n", encoding="utf-8")
    r = CliRunner().invoke(cli, ["plugin-quality-check", str(tmp_path), "--plugin-quality-config", str(cfg)])
    assert r.exit_code == 2 and "unknown rule" in r.output


def test_skill_validate_is_unchanged(tmp_path):
    # Same skill target, same score path: the skill score does not go through the plugin validator.
    skill = tmp_path / "good-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(GOOD_SKILL, encoding="utf-8")
    results = run_validation(skill, checks="quality", content_type="skill")
    scored = [r for r in results if r.metadata.get("quality_scores")]
    assert [r.validator_name for r in scored] == ["QUALITY"]
    assert not any(r.metadata.get("plugin_quality_scores") for r in results)
