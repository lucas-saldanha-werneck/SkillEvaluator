# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin quality score: a 0-100 score and A-F grade for a plugin's own parts.

Same scale as the skill quality score (four dimensions, skill weights, grade
cut-offs, ``--min-score``). Rules and points live in the packaged
``plugin-quality-default.yaml``; bundled skills keep their own skill score.
"""

from __future__ import annotations

import functools
from pathlib import Path

from skillevaluator.constants import QUALITY_DEFAULT_MIN_SCORE
from skillevaluator.models.quality import PluginQualityScoreResult
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_inventory import PluginInventory, plugin_inventory_for_root
from skillevaluator.plugin_quality_config import load_plugin_quality_config, load_plugin_quality_config_with_provenance
from skillevaluator.validators.base import ValidatorBase
from skillevaluator.validators.plugin_quality_rules import (
    build_context,
    check_correctness,
    check_discoverability,
    check_efficiency,
    check_reliability,
    component_counts,
    plugin_identity,
)
from skillevaluator.validators.quality_score import QualityScoreValidator


@functools.cache
def claude_only_rules() -> tuple[str, ...]:
    """Rule IDs that apply only to Claude Code plugins; the packaged config is read on first use, not at import."""
    return tuple(rule_id for rule_id, rule in load_plugin_quality_config().items() if rule["applies_to"] == "claude")


class PluginQualityScoreValidator(ValidatorBase):
    """Scores a plugin's manifest, commands, agents, hooks, mods and context cost on the skill scale."""

    def __init__(
        self,
        min_score: int = QUALITY_DEFAULT_MIN_SCORE,
        plugin_quality_config: Path | None = None,
        quality_config: Path | None = None,
        enforce: bool = False,
    ) -> None:
        self.min_score = min_score
        self.quality_config = quality_config
        self.enforce = enforce
        self.rules, self.provenance = load_plugin_quality_config_with_provenance(plugin_quality_config)

    @property
    def name(self) -> str:
        return "Plugin Quality Score (4-Dimension Analysis)"

    @property
    def description(self) -> str:
        return (
            "Plugin quality scoring across Correctness (35%), Discoverability (25%),"
            " Reliability (25%), and Efficiency (15%)"
        )

    def score(self, inventory: PluginInventory) -> PluginQualityScoreResult:
        """Score one plugin inventory; never raises on malformed plugin content."""
        ctx = build_context(inventory, tuple(self.rules["script_not_executable"].get("interpreters", ())))
        qs = PluginQualityScoreResult(
            plugin_name=plugin_identity(ctx) or inventory.root.name,
            manifest_type=inventory.selected_manifest.manifest_type,
        )
        check_correctness(ctx, qs, self.rules)
        check_discoverability(ctx, qs, self.rules)
        check_reliability(ctx, qs, self.rules)
        check_efficiency(ctx, qs, self.rules)
        if not ctx.claude:
            for rule_id in claude_only_rules():
                if rule_id not in qs.not_applicable:
                    qs.not_applicable.append(rule_id)
        qs.component_counts = component_counts(inventory)
        return qs

    def validate(self, plugin_path: Path) -> ValidationResult:
        """Score the plugin at ``plugin_path`` and its bundled skills.

        The plugin passes when its score is at least ``min_score``, no ``error``
        rule fired, and every bundled skill passes its own skill quality score.
        Without ``enforce`` the result never fails on the score: blocking issues
        are reported (``would_block``) but no gate fails.
        """
        result = ValidationResult(validator_name="PLUGIN_QUALITY", validator_description=self.description)
        result.metadata["plugin_quality_config"] = {**self.provenance, "min_score": self.min_score}
        inventory = plugin_inventory_for_root(plugin_path)
        if inventory is None:
            result.add_finding(
                Finding(
                    category="PLUGIN_QUALITY",
                    severity=Severity.HIGH if self.enforce else Severity.MEDIUM,
                    check_name="plugin_manifest_missing",
                    message="No supported plugin manifest could be safely located.",
                    file_path=str(plugin_path),
                    suggestion="Add .claude-plugin/plugin.json (or another supported manifest) at the plugin root.",
                )
            )
            return result

        qs = self.score(inventory)
        skills = self._score_skills(inventory)
        blocking: list[str] = []
        if qs.overall_score < self.min_score:
            blocking.append(f"score {qs.overall_score:.1f} < {self.min_score}")
        for rule_id in dict.fromkeys(issue.rule_id for issue in qs.all_issues if issue.severity == "error"):
            blocking.append(f"error rule: {rule_id}")
        for skill in skills:
            if not skill["passed"]:
                blocking.append(f"bundled skill '{skill.get('skill_name', '?')}' failed quality")

        manifest_file = inventory.selected_manifest.manifest_filename
        for issue in qs.all_issues:
            severity = _ENFORCED_SEVERITY if self.enforce else _ADVISORY_SEVERITY
            metadata = {
                "dimension": issue.dimension,
                "deduction": issue.deduction,
                "rule_id": issue.rule_id,
                "source": self.rules[issue.rule_id]["source"] if issue.rule_id in self.rules else None,
                "line": issue.line,
            }
            if not self.enforce and issue.severity == "error":
                metadata["would_block"] = True
            result.add_finding(
                Finding(
                    category="PLUGIN_QUALITY",
                    severity=severity.get(issue.severity, Severity.LOW),
                    check_name=f"plugin_quality_{issue.dimension}",
                    message=f"[{issue.rule_id}] {issue.message}" if issue.rule_id else issue.message,
                    file_path=issue.file or manifest_file,
                    line_number=issue.line,
                    suggestion=issue.suggestion,
                    metadata=metadata,
                )
            )
        if self.enforce:
            for reason in blocking:
                result.fail_gate(f"Plugin quality: {reason}")

        result.metadata["plugin_quality_scores"] = {
            **qs.to_dict(),
            "min_score": self.min_score,
            "enforced": self.enforce,
            "passed": not blocking,
            "blocking": blocking,
            "skill_count": len(skills),
            "skills": skills,
        }
        result.add_success(
            check_name="plugin_quality_score",
            message=f"Score: {qs.overall_score:.1f}/100 (Grade: {qs.grade})",
            overall_score=round(qs.overall_score, 1),
            grade=qs.grade,
        )
        return result

    def _score_skills(self, inventory: PluginInventory) -> list[dict]:
        """Each bundled skill's own quality summary; its findings are not merged, so advisory mode cannot fail."""
        validator = QualityScoreValidator(min_score=self.min_score, quality_config=self.quality_config)
        skills: list[dict] = []
        for member in inventory.selected_skills:
            sub = validator.validate(inventory.root / str(member.path))
            summary = dict(sub.metadata.get("quality_scores") or {"skill_name": member.name})
            summary["passed"] = bool(sub.passed)
            skills.append(summary)
        return skills


_ENFORCED_SEVERITY = {"error": Severity.HIGH, "warning": Severity.MEDIUM, "info": Severity.LOW}
_ADVISORY_SEVERITY = {"error": Severity.MEDIUM, "warning": Severity.MEDIUM, "info": Severity.LOW}
