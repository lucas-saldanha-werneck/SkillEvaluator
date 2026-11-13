# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin quality rules: the static context a plugin is scored on and the four dimension checks.

Every rule is named in ``config/plugin-quality-default.yaml`` (points, severity,
scope, parameters, source). Checks only read files; no plugin code runs.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES, PLUGIN_CONTAINED_MANIFEST_TYPE
from skillevaluator.plugin_component_risk import iter_hook_handlers
from skillevaluator.plugin_format_schema import ManifestIssue, validate_manifest_fields
from skillevaluator.plugin_inventory import _finding_rel_path
from skillevaluator.plugin_manifest import PluginManifestPathError, find_claude_marketplace_entry
from skillevaluator.plugin_mods import ModInventory, read_mods
from skillevaluator.utils.secure_fs import SecurePathError

if TYPE_CHECKING:
    from skillevaluator.models.quality import PluginQualityScoreResult
    from skillevaluator.plugin_inventory import PluginInventory
    from skillevaluator.plugin_quality_config import PluginQualityRule

_NAME_RE = re.compile(r"^[a-z0-9-]+$")
_PLUGIN_ROOT_PREFIXES = ("${CLAUDE_PLUGIN_ROOT}/", "$CLAUDE_PLUGIN_ROOT/")
_SCRIPT_SUFFIXES = (".sh", ".bash", ".zsh", ".py", ".js", ".mjs", ".cjs", ".ts", ".mts")
_SHELL_STATE_RE = re.compile(
    r"(>>?\s*|\btee\s+(-a\s+)?|\bmkdir\s+(-p\s+)?|\btouch\s+|\bcp\s+\S+\s+|\bmv\s+\S+\s+)[\"']?\$\{?CLAUDE_PLUGIN_ROOT\b"
)
_CODE_WRITE_RE = re.compile(
    r"open\([^)]*,\s*(?:mode\s*=\s*)?['\"][^'\"]*[wax+][^'\"]*['\"]|\bwrite(?:File|FileSync)\b|"
    r"\bappend(?:File|FileSync)\b|\.write_(?:text|bytes)\(|\bcreateWriteStream\b"
)
_TOKEN_TRAILING = ";&|)"
_MAX_LISTED = 3


@dataclass
class HookHandlerView:
    """One settings-hook handler with where it came from."""

    event: str
    matcher: str | None
    handler: dict
    file: str
    inline: bool
    config_has_description: bool


@dataclass
class PluginQualityContext:
    """Everything the plugin rules read, gathered once from the inventory."""

    inventory: PluginInventory
    root: Path
    manifest: dict
    manifest_issues: list[ManifestIssue]
    manifest_readable: bool
    marketplace_entry: dict | None
    claude: bool
    handlers: list[HookHandlerView] = field(default_factory=list)
    mods: list[ModInventory] = field(default_factory=list)
    readme: str | None = None
    source_files: tuple[PurePosixPath, ...] = ()
    interpreters: tuple[str, ...] = ()


def _read_readme(inventory: PluginInventory) -> str | None:
    for path in inventory.source_files:
        if len(path.parts) == 1 and path.name.lower() == "readme.md":
            try:
                return inventory._reader.read_text(path, CONTENT_DEDUP_MAX_FILE_BYTES)
            except (SecurePathError, OSError, UnicodeError, ValueError):
                return ""  # present but unreadable
    return None


def build_context(inventory: PluginInventory, interpreters: tuple[str, ...] = ()) -> PluginQualityContext:
    """Collect the manifest, marketplace entry, hook handlers, mods and README of one plugin."""
    selected = inventory.selected_manifest
    try:
        parsed = selected.parse_for_audit()
    except (PluginManifestPathError, SecurePathError, OSError, ValueError):
        # an unsafe manifest is scored as unreadable; Tier 1 fails it separately
        parsed = None
    readable = isinstance(parsed, dict)
    manifest: dict = parsed if isinstance(parsed, dict) else {}
    issues = validate_manifest_fields(selected.manifest_type, manifest) if readable else []
    name = manifest.get("name") if isinstance(manifest.get("name"), str) else None
    entry = find_claude_marketplace_entry(inventory.root, plugin_name=name, stop=None)
    claude = selected.manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE
    handlers: list[HookHandlerView] = []
    hook_members = [m for m in inventory.selected_components if m.type == "hook"]
    for member in hook_members:
        config = member.config
        has_description = isinstance(config, dict) and isinstance(config.get("description"), str)
        inline = member.path == selected.manifest_filename
        for event, _group, matcher, _index, handler in iter_hook_handlers(config):
            if isinstance(handler, dict):
                handlers.append(
                    HookHandlerView(event, matcher, handler, str(member.path or ""), inline, has_description)
                )
    mods = read_mods(inventory.root, hook_members, inventory.source_files) if claude else []
    return PluginQualityContext(
        inventory=inventory,
        root=inventory.root,
        manifest=manifest,
        manifest_issues=issues,
        manifest_readable=readable,
        marketplace_entry=entry,
        claude=claude,
        handlers=handlers,
        mods=mods,
        readme=_read_readme(inventory),
        source_files=tuple(inventory.source_files),
        interpreters=tuple(interpreters),
    )


def apply_rule(
    qs: PluginQualityScoreResult,
    rules: dict[str, PluginQualityRule],
    rule_id: str,
    detail: str = "",
    *,
    ctx: PluginQualityContext,
    file: str | None = None,
    line: int | None = None,
    count: int = 1,
) -> None:
    """Deduct one rule's points (``count`` times points, capped by ``cap``) from its dimension."""
    rule = rules[rule_id]
    if rule["applies_to"] == "claude" and not ctx.claude:
        if rule_id not in qs.not_applicable:
            qs.not_applicable.append(rule_id)
        return
    points = rule["points"]
    if not points or count <= 0:
        return
    amount = count * points
    if "cap" in rule:
        amount = min(amount, rule["cap"])
    message = f"{rule['message']}: {detail}" if detail else rule["message"]
    dimension = getattr(qs, rule["dimension"])
    dimension.deduct(amount, rule["severity"], message, rule["suggestion"], rule_id=rule_id, file=file, line=line)


def _listed(items: list[str]) -> str:
    unique = list(dict.fromkeys(items))
    shown = ", ".join(unique[:_MAX_LISTED])
    return shown + (f" (+{len(unique) - _MAX_LISTED} more)" if len(unique) > _MAX_LISTED else "")


def _name_has_term(name: str, terms: list[str]) -> bool:
    """Reserved term as a name segment; unlike prose, ``_`` also separates segments (``Claude_Tool``)."""
    return any(re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", name, re.IGNORECASE) for term in terms)


def _manifest_file(ctx: PluginQualityContext) -> str:
    return ctx.inventory.selected_manifest.manifest_filename


def _finding_file(ctx: PluginQualityContext, file_path: str) -> str | None:
    return _finding_rel_path(file_path, ctx.root)


# --------------------------------------------------------------------------- #
# Correctness                                                                 #
# --------------------------------------------------------------------------- #


def _check_manifest(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    manifest_file = _manifest_file(ctx)
    if not ctx.manifest_readable:
        apply_rule(qs, rules, "manifest_invalid", "manifest cannot be read or parsed", ctx=ctx, file=manifest_file)
    else:
        errors = [issue for issue in ctx.manifest_issues if issue.level == "error"]
        if errors:
            apply_rule(
                qs,
                rules,
                "manifest_invalid",
                _listed([f"{issue.field}: {issue.message}" for issue in errors]),
                ctx=ctx,
                file=manifest_file,
            )

    findings = ctx.inventory.findings
    missing = [f for f in findings if f.check_name == "plugin_component_path_missing"]
    if missing:
        apply_rule(
            qs,
            rules,
            "component_path_missing",
            _listed([str(f.metadata.get("plugin_component_ref", f.message)) for f in missing]),
            ctx=ctx,
            file=_finding_file(ctx, missing[0].file_path),
        )
    outside = [f for f in findings if f.check_name in {"plugin_component_path_escape", "plugin_component_path_style"}]
    if outside:
        apply_rule(
            qs,
            rules,
            "component_path_outside_root",
            _listed([str(f.metadata.get("plugin_component_ref", f.message)) for f in outside]),
            ctx=ctx,
            file=_finding_file(ctx, outside[0].file_path),
        )

    name = ctx.manifest.get("name")
    schema_flagged_name = any(issue.field == "name" and issue.level == "error" for issue in ctx.manifest_issues)
    if isinstance(name, str) and name:
        if not _NAME_RE.match(name) and not schema_flagged_name:
            apply_rule(qs, rules, "name_format", repr(name), ctx=ctx, file=manifest_file)
        if _name_has_term(name, rules["name_reserved"].get("terms", [])):
            apply_rule(qs, rules, "name_reserved", repr(name), ctx=ctx, file=manifest_file)

    entry = ctx.marketplace_entry or {}
    if not ctx.manifest.get("author") and not entry.get("author"):
        apply_rule(qs, rules, "author_missing", ctx=ctx, file=manifest_file)
    if not ctx.manifest.get("version") and not entry.get("version"):
        apply_rule(qs, rules, "version_missing", ctx=ctx, file=manifest_file)


def hook_command_scripts(command: str, ctx: PluginQualityContext) -> list[tuple[PurePosixPath, bool, bool]]:
    """Plugin files a hook command names: ``(path, via_plugin_root, has_interpreter)``.

    A token counts when, after an optional ``${CLAUDE_PLUGIN_ROOT}/`` prefix, it
    names a file in the plugin's source tree. ``has_interpreter`` is true when
    the command's first word (its basename) is a listed interpreter.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return []
    interpreters = set(ctx.interpreters)
    has_interpreter = posixpath.basename(tokens[0]) in interpreters
    files = set(ctx.source_files)
    found: list[tuple[PurePosixPath, bool, bool]] = []
    for token in tokens:
        token = token.rstrip(_TOKEN_TRAILING)
        via_root = token.startswith(_PLUGIN_ROOT_PREFIXES)
        if via_root:
            token = token.split("/", 1)[1]
        elif "$" in token or token.startswith(("/", "~")):
            continue
        rel = posixpath.normpath(token)
        if rel.startswith("../") or rel in {".", ".."}:
            continue
        path = PurePosixPath(rel)
        if path in files and (path, via_root, has_interpreter) not in found:
            found.append((path, via_root, has_interpreter))
    return found


def _command_handlers(ctx: PluginQualityContext) -> list[HookHandlerView]:
    return [
        view
        for view in ctx.handlers
        if view.handler.get("type") == "command" and isinstance(view.handler.get("command"), str)
    ]


def _hook_scripts(ctx: PluginQualityContext) -> list[tuple[PurePosixPath, bool, bool]]:
    scripts: list[tuple[PurePosixPath, bool, bool]] = []
    for view in _command_handlers(ctx):
        for item in hook_command_scripts(view.handler["command"], ctx):
            if item not in scripts:
                scripts.append(item)
    return scripts


def read_plugin_text(ctx: PluginQualityContext, path: PurePosixPath) -> str | None:
    try:
        return ctx.inventory._reader.read_text(path, CONTENT_DEDUP_MAX_FILE_BYTES)
    except (SecurePathError, OSError, UnicodeError, ValueError):
        return None


def _state_write_line(text: str, suffix: str) -> int | None:
    for number, line in enumerate(text.splitlines(), start=1):
        if suffix in {".sh", ".bash", ".zsh", ""}:
            if _SHELL_STATE_RE.search(line):
                return number
        elif "CLAUDE_PLUGIN_ROOT" in line and _CODE_WRITE_RE.search(line):
            return number
    return None


def _check_hooks_and_mods(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    selected = ctx.inventory.selected_manifest.manifest_filename
    for member in ctx.inventory.selected_components:
        if member.type != "hook" or member.path is None or member.path == selected:
            continue
        config = member.config
        if isinstance(config, dict) and "hooks" not in config and "modules" not in config:
            apply_rule(qs, rules, "hooks_file_unwrapped", str(member.path), ctx=ctx, file=str(member.path))

    unknown = [
        record
        for record in ctx.inventory.hook_records
        if "unknown_event" in record.risk_flags or record.handler_type == "invalid"
    ]
    if unknown:
        apply_rule(
            qs,
            rules,
            "hook_event_unknown",
            _listed([f"{r.event} ({r.handler_type})" if r.handler_type == "invalid" else r.event for r in unknown]),
            ctx=ctx,
            file=unknown[0].file or None,
        )

    relative = [path.as_posix() for path, via_root, _interp in _hook_scripts(ctx) if not via_root]
    if relative:
        first = next(v for v in _command_handlers(ctx) if any(r in v.handler["command"] for r in relative))
        apply_rule(qs, rules, "hook_without_plugin_root", _listed(relative), ctx=ctx, file=first.file)

    writes: list[tuple[str, int]] = []
    for view in _command_handlers(ctx):
        line = _state_write_line(view.handler["command"], "")
        if line is not None:
            writes.append((view.file, 0))
    for path, _via_root, _interp in _hook_scripts(ctx):
        if path.suffix not in _SCRIPT_SUFFIXES:
            continue
        text = read_plugin_text(ctx, path)
        line = _state_write_line(text, path.suffix) if text is not None else None
        if line is not None:
            writes.append((path.as_posix(), line))
    if writes:
        file, line = writes[0]
        apply_rule(
            qs,
            rules,
            "state_in_plugin_root",
            _listed([f for f, _ in writes]),
            ctx=ctx,
            file=file,
            line=line or None,
        )

    problems = [(mod.hooks_file, raw, reason) for mod in ctx.mods for raw, reason in mod.module_problems]
    if problems:
        apply_rule(
            qs,
            rules,
            "mod_module_invalid",
            _listed([f"{raw} ({reason})" for _file, raw, reason in problems]),
            ctx=ctx,
            file=problems[0][0],
        )


def check_correctness(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    _check_manifest(ctx, qs, rules)
    _check_hooks_and_mods(ctx, qs, rules)


def check_discoverability(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    return None


def check_reliability(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    return None


def check_efficiency(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    return None


def component_counts(inventory: PluginInventory) -> dict[str, Any]:
    counts = inventory.component_inventory().get("counts", {})
    return dict(counts) if isinstance(counts, dict) else {}
