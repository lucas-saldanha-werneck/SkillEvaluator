# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin quality score: a 0-100 score and A-F grade for a plugin's own parts.

Same scale as the skill quality score (four dimensions, skill weights, grade
cut-offs, ``--min-score``). Rules and points live in the packaged
``plugin-quality-default.yaml``; bundled skills keep their own skill score.
"""

from __future__ import annotations

from pathlib import Path

from skillevaluator.constants import QUALITY_DEFAULT_MIN_SCORE
from skillevaluator.models.quality import PluginQualityScoreResult
from skillevaluator.models.result import ValidationResult
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
)

CLAUDE_ONLY_RULES: tuple[str, ...] = tuple(
    rule_id for rule_id, rule in load_plugin_quality_config().items() if rule["applies_to"] == "claude"
)


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
        ctx = build_context(inventory)
        name = ctx.manifest.get("name")
        qs = PluginQualityScoreResult(
            plugin_name=name if isinstance(name, str) else inventory.root.name,
            manifest_type=inventory.selected_manifest.manifest_type,
        )
        check_correctness(ctx, qs, self.rules)
        check_discoverability(ctx, qs, self.rules)
        check_reliability(ctx, qs, self.rules)
        check_efficiency(ctx, qs, self.rules)
        if not ctx.claude:
            for rule_id in CLAUDE_ONLY_RULES:
                if rule_id not in qs.not_applicable:
                    qs.not_applicable.append(rule_id)
        qs.component_counts = component_counts(inventory)
        return qs

    def validate(self, plugin_path: Path) -> ValidationResult:
        result = ValidationResult(validator_name="PLUGIN_QUALITY", validator_description=self.description)
        inventory = plugin_inventory_for_root(plugin_path)
        if inventory is not None:
            result.metadata["plugin_quality_scores"] = self.score(inventory).to_dict()
        return result
