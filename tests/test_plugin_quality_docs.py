# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plugin quality score page documents every packaged rule and is published."""

from __future__ import annotations

from pathlib import Path

from skillevaluator.plugin_quality_config import load_plugin_quality_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_plugin_rule_is_documented():
    page = (REPO_ROOT / "docs" / "plugin-quality-score.mdx").read_text(encoding="utf-8")
    for rule_id, rule in load_plugin_quality_config().items():
        assert f"`{rule_id}`" in page, rule_id
        assert rule["source"] in page, rule_id


def test_page_is_registered_in_fern():
    assert "../docs/plugin-quality-score.mdx" in (REPO_ROOT / "fern" / "docs.yml").read_text(encoding="utf-8")
