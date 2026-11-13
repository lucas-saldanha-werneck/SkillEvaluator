# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixture builders for plugin quality score tests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

GOOD_MANIFEST: dict = {
    "name": "table-tools",
    "version": "1.0.0",
    "author": {"name": "Example Author"},
    "description": "Formats Markdown tables in pull request descriptions and checks their column widths.",
}


def make_plugin(
    root: Path,
    *,
    manifest: dict | str | None,
    files: Mapping[str, str] | None = None,
    executable: Iterable[str] = (),
) -> Path:
    """Write a plugin tree: ``.claude-plugin/plugin.json``, extra files, and executable bits.

    ``manifest`` as a dict is written as JSON, as a str verbatim (for invalid JSON),
    and ``None`` writes no manifest.
    """
    root.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (root / ".claude-plugin").mkdir(exist_ok=True)
        text = manifest if isinstance(manifest, str) else json.dumps(manifest)
        (root / ".claude-plugin" / "plugin.json").write_text(text, encoding="utf-8")
    for relative, content in (files or {}).items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for relative in executable:
        (root / relative).chmod(0o755)
    return root
