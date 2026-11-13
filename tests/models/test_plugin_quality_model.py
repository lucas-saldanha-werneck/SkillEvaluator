# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plugin quality score model and the optional issue fields."""

from __future__ import annotations

from skillevaluator.models.quality import PluginQualityScoreResult, QualityDimension


def test_skill_issue_dict_is_unchanged_without_rule_fields():
    dim = QualityDimension(name="correctness", weight=0.35)
    dim.deduct(5, "info", "No examples provided", "Add example usage")
    assert set(dim.to_dict()["issues"][0]) == {"severity", "message", "suggestion", "deduction"}


def test_rule_fields_are_serialized_when_set():
    dim = QualityDimension(name="reliability", weight=0.25)
    dim.deduct(5, "info", "msg", rule_id="hook_timeout_missing", file="hooks/hooks.json", line=4)
    issue = dim.to_dict()["issues"][0]
    assert (issue["rule_id"], issue["file"], issue["line"]) == ("hook_timeout_missing", "hooks/hooks.json", 4)


def test_plugin_result_uses_skill_weights_and_grades():
    qs = PluginQualityScoreResult(plugin_name="demo", manifest_type="claude_plugin_json")
    qs.correctness.deduct(20, "warning", "x")  # 80 * .35 = 28
    assert qs.overall_score == 28 + 25 + 25 + 15  # 93.0
    assert qs.grade == "A"
    d = qs.to_dict()
    assert d["dimensions"]["correctness"]["score"] == 80.0
    assert d["dimensions"]["efficiency"]["weight"] == 0.15
    assert d["not_applicable"] == [] and d["metrics"]["always_on_tokens"] == 0
