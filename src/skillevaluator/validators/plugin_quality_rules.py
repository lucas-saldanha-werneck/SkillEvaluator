# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin quality rules: the static context a plugin is scored on and the four dimension checks.

Every rule is named in ``config/plugin-quality-default.yaml`` (points, severity,
scope, parameters, source). Checks only read files; no plugin code runs.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_OPENCLAW_MANIFEST_TYPE,
    QUALITY_PERSON_PHRASES,
    QUALITY_VAGUE_WORDS,
)
from skillevaluator.plugin_component_risk import iter_hook_handlers
from skillevaluator.plugin_format_schema import ManifestIssue, validate_manifest_fields
from skillevaluator.plugin_formats import manifest_identity
from skillevaluator.plugin_inventory import _finding_rel_path
from skillevaluator.plugin_manifest import PluginManifestPathError, find_claude_marketplace_entry
from skillevaluator.plugin_markdown import parse_markdown
from skillevaluator.plugin_mods import ModInventory, mod_test_files, read_mods
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.validators.quality_score import _contains_any_term

if TYPE_CHECKING:
    from skillevaluator.models.quality import PluginQualityScoreResult
    from skillevaluator.plugin_inventory import PluginInventory
    from skillevaluator.plugin_quality_config import PluginQualityRule

_NAME_RE = re.compile(r"^[a-z0-9-]+$")
_PLUGIN_ROOT_PREFIXES = ("${CLAUDE_PLUGIN_ROOT}/", "$CLAUDE_PLUGIN_ROOT/")
_PLUGIN_ROOT_BARE = ("${CLAUDE_PLUGIN_ROOT}", "$CLAUDE_PLUGIN_ROOT")
_SCRIPT_SUFFIXES = (".sh", ".bash", ".zsh", ".py", ".js", ".mjs", ".cjs", ".ts", ".mts")
_SHELL_STATE_RE = re.compile(
    r"(>>?\s*|\btee\s+(-a\s+)?|\bmkdir\s+(-p\s+)?|\btouch\s+|\bcp\s+\S+\s+|\bmv\s+\S+\s+)[\"']?\$\{?CLAUDE_PLUGIN_ROOT\b"
)
# Formats whose manifest schema has no ``author`` field (OpenClaw: docs.openclaw.ai/plugins/manifest).
_NO_AUTHOR_MANIFEST_TYPES = frozenset({PLUGIN_OPENCLAW_MANIFEST_TYPE})
_PLUGIN_ROOT_VAR = "CLAUDE_PLUGIN_ROOT"
_OPEN_CALL_RE = re.compile(r"(?<![\w.])open\(")
_WRITE_CALL_RE = re.compile(r"\b(?:writeFile|writeFileSync|appendFile|appendFileSync|createWriteStream)\(")
_WRITE_METHOD_RE = re.compile(r"\.write_(?:text|bytes)\(")
_WRITE_MODE_RE = re.compile(r"^[rwaxbt+]+$")
_MATCH_ALL_MATCHERS = frozenset({None, "", "*", ".*"})
_TOKEN_TRAILING = ";&|)"
_MAX_LISTED = 3


def _call_args(text: str, start: int) -> list[str]:
    """Top-level arguments of the call whose ``(`` ends at ``start``; an unclosed call runs to the end of ``text``."""
    args: list[str] = []
    depth = 0
    quote = ""
    current = start
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if char == quote and text[index - 1] != "\\":
                quote = ""
        elif char in "'\"`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                args.append(text[current:index])
                return args
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[current:index])
            current = index + 1
    args.append(text[current:])
    return args


def _open_mode_writes(args: list[str]) -> bool:
    mode = next((a.strip()[5:].strip() for a in args[1:] if a.strip().startswith("mode=")), None)
    if mode is None and len(args) > 1:
        mode = args[1].strip()
    literal = mode[1:-1] if mode and len(mode) >= 2 and mode[0] == mode[-1] and mode[0] in "'\"" else None
    return bool(literal) and _WRITE_MODE_RE.match(literal) is not None and any(c in literal for c in "wax+")


def _writes_under_plugin_root(line: str) -> bool:
    """Whether one source line writes a file whose destination mentions ``CLAUDE_PLUGIN_ROOT``.

    Reads (``open(path)``, ``open(path, 'r')``) and writes to another directory that merely
    mention the root elsewhere on the line do not count.
    """
    if _PLUGIN_ROOT_VAR not in line:
        return False
    for match in _OPEN_CALL_RE.finditer(line):
        args = _call_args(line, match.end())
        if _PLUGIN_ROOT_VAR in args[0] and _open_mode_writes(args):
            return True
    for match in _WRITE_CALL_RE.finditer(line):
        if _PLUGIN_ROOT_VAR in _call_args(line, match.end())[0]:
            return True
    return any(_PLUGIN_ROOT_VAR in line[: match.start()] for match in _WRITE_METHOD_RE.finditer(line))


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
    identity = manifest_identity(selected.manifest_type, manifest)
    entry = find_claude_marketplace_entry(
        inventory.root, plugin_name=identity if isinstance(identity, str) else None, stop=None
    )
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


def plugin_identity(ctx: PluginQualityContext) -> str | None:
    """The plugin's declared identity (OpenClaw ``id``, otherwise ``name``), or None when absent or not a string."""
    value = manifest_identity(ctx.inventory.selected_manifest.manifest_type, ctx.manifest)
    return value if isinstance(value, str) and value else None


def _readme_file(ctx: PluginQualityContext) -> str | None:
    return next((p.as_posix() for p in ctx.source_files if len(p.parts) == 1 and p.name.lower() == "readme.md"), None)


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

    name = plugin_identity(ctx)
    identity_field = "id" if ctx.inventory.selected_manifest.manifest_type == PLUGIN_OPENCLAW_MANIFEST_TYPE else "name"
    schema_flagged_name = any(issue.field == identity_field and issue.level == "error" for issue in ctx.manifest_issues)
    if isinstance(name, str) and name:
        if not _NAME_RE.match(name) and not schema_flagged_name:
            apply_rule(qs, rules, "name_format", repr(name), ctx=ctx, file=manifest_file)
        if _name_has_term(name, rules["name_reserved"].get("terms", [])):
            apply_rule(qs, rules, "name_reserved", repr(name), ctx=ctx, file=manifest_file)

    entry = ctx.marketplace_entry or {}
    if ctx.inventory.selected_manifest.manifest_type in _NO_AUTHOR_MANIFEST_TYPES:
        if "author_missing" not in qs.not_applicable:
            qs.not_applicable.append("author_missing")
    elif not ctx.manifest.get("author") and not entry.get("author"):
        apply_rule(qs, rules, "author_missing", ctx=ctx, file=manifest_file)
    if not ctx.manifest.get("version") and not entry.get("version"):
        apply_rule(qs, rules, "version_missing", ctx=ctx, file=manifest_file)


def hook_command_scripts(command: str, ctx: PluginQualityContext) -> list[tuple[PurePosixPath, bool, bool]]:
    """Plugin files a hook command names: ``(path, via_plugin_root, has_interpreter)``.

    A token counts when, after an optional ``${CLAUDE_PLUGIN_ROOT}/`` prefix, it
    names a file in the plugin's source tree. ``has_interpreter`` is true when
    the command's first word (its basename) is a listed interpreter.
    """
    return _scripts_from_tokens(_split(command), ctx)


def _split(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def handler_tokens(handler: dict) -> list[str]:
    """The argument vector a command hook runs.

    Exec form (``args`` present) spawns ``command`` directly with ``args``; shell form
    splits ``command`` like a shell would.
    """
    command = handler.get("command")
    if not isinstance(command, str):
        return []
    args = handler.get("args")
    if isinstance(args, list):
        return [command, *(str(arg) for arg in args)]
    return _split(command)


def hook_handler_scripts(handler: dict, ctx: PluginQualityContext) -> list[tuple[PurePosixPath, bool, bool]]:
    """:func:`hook_command_scripts` for a handler, in shell form or exec form (``command`` + ``args``)."""
    return _scripts_from_tokens(handler_tokens(handler), ctx)


def _scripts_from_tokens(tokens: list[str], ctx: PluginQualityContext) -> list[tuple[PurePosixPath, bool, bool]]:
    if not tokens:
        return []
    interpreters = set(ctx.interpreters)
    has_interpreter = posixpath.basename(tokens[0]) in interpreters
    files = set(ctx.source_files)
    found: list[tuple[PurePosixPath, bool, bool]] = []
    cd_dir: str | None = None  # plugin-relative directory after ``cd ${CLAUDE_PLUGIN_ROOT}[/sub]``
    for index, token in enumerate(tokens):
        if token == "cd" and index + 1 < len(tokens):
            target = tokens[index + 1].rstrip(_TOKEN_TRAILING)
            if target in _PLUGIN_ROOT_BARE:
                cd_dir = "."
            elif target.startswith(_PLUGIN_ROOT_PREFIXES):
                cd_dir = posixpath.normpath(target.split("/", 1)[1])
            continue
        token = token.rstrip(_TOKEN_TRAILING)
        via_root = token.startswith(_PLUGIN_ROOT_PREFIXES)
        if via_root:
            token = token.split("/", 1)[1]
        elif "$" in token or token.startswith(("/", "~")):
            continue
        elif cd_dir is not None:
            token = posixpath.join(cd_dir, token)
            via_root = True
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
        for item in hook_handler_scripts(view.handler, ctx):
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
        elif _writes_under_plugin_root(line):
            return number
    return None


def _check_hooks_and_mods(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    selected = ctx.inventory.selected_manifest.manifest_filename
    for member in ctx.inventory.selected_components:
        if member.type != "hook" or member.path is None or member.path == selected:
            continue
        if "#hooks" in str(member.path):  # SKILL.md / command frontmatter: the event map itself, no envelope
            continue
        config = member.config
        if isinstance(config, dict) and "hooks" not in config and "modules" not in config:
            apply_rule(qs, rules, "hooks_file_unwrapped", str(member.path), ctx=ctx, file=str(member.path))

    unknown = [
        record
        for record in ctx.inventory.hook_records
        if "unknown_event" in record.risk_flags or record.handler_type in {"invalid", "unknown"}
    ]
    if unknown:
        apply_rule(
            qs,
            rules,
            "hook_event_unknown",
            _listed(
                [
                    f"{r.event} ({r.handler_type})" if r.handler_type in {"invalid", "unknown"} else r.event
                    for r in unknown
                ]
            ),
            ctx=ctx,
            file=unknown[0].file or None,
        )

    relative: list[str] = []
    first_file: str | None = None
    for view in _command_handlers(ctx):
        for path, via_root, _interp in hook_handler_scripts(view.handler, ctx):
            if not via_root:
                relative.append(path.as_posix())
                first_file = first_file or view.file
    if relative:
        apply_rule(qs, rules, "hook_without_plugin_root", _listed(relative), ctx=ctx, file=first_file)

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


_MIN_FIRST_LINE_CHARS = 10


def _first_body_line(member: Any) -> str:
    """First non-empty line of a command's Markdown body, the description Claude Code falls back to.

    Documented for skills, and commands are skills (code.claude.com/docs/en/skills, "description": "If omitted,
    uses the first non-empty line of the markdown content"). A code fence or a line shorter than
    ``_MIN_FIRST_LINE_CHARS`` is not a useful description.
    """
    if member.type != "command" or not isinstance(member.text, str):
        return ""
    body = parse_markdown(member.text).body
    line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    if line.startswith(("```", "~~~")):
        return ""
    line = line.lstrip("#").strip()
    return line if len(line) >= _MIN_FIRST_LINE_CHARS else ""


def _component_description(member: Any) -> str:
    if member.path is None and isinstance(member.config, dict):  # command map entry
        value = member.config.get("description")
    else:
        value = member.frontmatter.get("description") if isinstance(member.frontmatter, dict) else None
        if not value and isinstance(member.config, dict):
            value = member.config.get("description")
    described = value.strip() if isinstance(value, str) else ""
    return described or _first_body_line(member)


def check_discoverability(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    manifest_file = _manifest_file(ctx)
    raw = ctx.manifest.get("description")
    desc = raw.strip() if isinstance(raw, str) else ""
    min_chars = rules["description_short"].get("min_chars", 20)
    if not desc:
        apply_rule(qs, rules, "description_short", "missing", ctx=ctx, file=manifest_file)
    else:
        if len(desc) < min_chars:
            apply_rule(
                qs, rules, "description_short", f"{len(desc)} < {min_chars} characters", ctx=ctx, file=manifest_file
            )
        max_chars = rules["description_long"].get("max_chars", 200)
        if len(desc) > max_chars:
            apply_rule(
                qs, rules, "description_long", f"{len(desc)} > {max_chars} characters", ctx=ctx, file=manifest_file
            )
        if _contains_any_term(desc, QUALITY_VAGUE_WORDS):
            apply_rule(qs, rules, "description_vague", ctx=ctx, file=manifest_file)
        if _contains_any_term(desc, QUALITY_PERSON_PHRASES):
            apply_rule(qs, rules, "description_person", ctx=ctx, file=manifest_file)

    undescribed = [
        member
        for member in ctx.inventory.selected_components
        if member.type in {"command", "agent"} and not member.problem and not _component_description(member)
    ]
    if undescribed:
        apply_rule(
            qs,
            rules,
            "component_description_missing",
            _listed([f"{m.type} {m.name}" for m in undescribed]),
            ctx=ctx,
            file=undescribed[0].path or manifest_file,
            count=len(undescribed),
        )

    reg_min = rules["mod_registration_description"].get("min_chars", 20)
    weak = []
    for mod in ctx.mods:
        for registration in mod.registrations:
            if not registration.resolved:
                qs.unresolved_registrations += 1
            elif registration.description is None or len(registration.description.strip()) < reg_min:
                weak.append(registration)
    if weak:
        apply_rule(
            qs,
            rules,
            "mod_registration_description",
            _listed([f"{r.kind} {r.name or '?'}" for r in weak]),
            ctx=ctx,
            file=weak[0].file,
            line=weak[0].line,
            count=len(weak),
        )

    if ctx.readme is None:
        apply_rule(qs, rules, "readme_missing", ctx=ctx)

    silent = [
        view for view in ctx.handlers if not view.config_has_description and not view.handler.get("statusMessage")
    ]
    if silent:
        apply_rule(
            qs,
            rules,
            "hook_status_message_missing",
            _listed([f"{view.file}:{view.event}" for view in silent]),
            ctx=ctx,
            file=silent[0].file,
        )


_MARKER_LANGUAGE = {
    ".sh": "sh",
    ".bash": "sh",
    ".zsh": "sh",
    ".py": "py",
    ".js": "js",
    ".mjs": "js",
    ".cjs": "js",
    ".ts": "js",
    ".mts": "js",
}


def _script_language(path: PurePosixPath, text: str) -> str | None:
    if path.suffix:
        return _MARKER_LANGUAGE.get(path.suffix)
    first = text.split("\n", 1)[0]
    if first.startswith("#!"):  # extensionless script: read the interpreter from its shebang
        if re.search(r"\b(?:ba|z)?sh\b", first):
            return "sh"
        if "python" in first:
            return "py"
        if re.search(r"\b(?:node|bun|deno)\b", first):
            return "js"
    return None


def _direct_scripts(ctx: PluginQualityContext) -> list[PurePosixPath]:
    """Plugin files a hook command runs as its first word, without an interpreter in front."""
    direct: list[PurePosixPath] = []
    for view in _command_handlers(ctx):
        first = handler_tokens(view.handler)[:1]
        if not first:
            continue
        for path, _via_root, has_interpreter in hook_command_scripts(first[0], ctx):
            if not has_interpreter and path not in direct:
                direct.append(path)
    return direct


def _matches_test_pattern(path: PurePosixPath, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            if pattern.rstrip("/") in path.parts[:-1]:
                return True
        elif fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def check_reliability(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    markers = rules["hook_script_error_handling"].get("markers", {})
    unhandled: list[str] = []
    executed = [path for path, _via_root, _interp in _hook_scripts(ctx) if path.suffix in _SCRIPT_SUFFIXES]
    executed += [path for path in _direct_scripts(ctx) if path not in executed]
    for path in executed:
        text = read_plugin_text(ctx, path)
        if text is None:
            continue
        language = _script_language(path, text)
        if language is None or language not in markers:
            continue
        handled = any(marker in text for marker in markers[language])
        if language == "py" and "try:" in text and "except" in text:
            handled = True
        if not handled:
            unhandled.append(path.as_posix())
    if unhandled:
        apply_rule(qs, rules, "hook_script_error_handling", _listed(unhandled), ctx=ctx, file=unhandled[0])

    no_timeout = [view for view in _command_handlers(ctx) if "timeout" not in view.handler]
    if no_timeout:
        apply_rule(
            qs,
            rules,
            "hook_timeout_missing",
            _listed([f"{view.file}:{view.event}" for view in no_timeout]),
            ctx=ctx,
            file=no_timeout[0].file,
        )

    not_executable: list[str] = []
    for path in _direct_scripts(ctx):
        try:
            mode = (ctx.root / path).lstat().st_mode
        except OSError:
            continue
        if not mode & 0o111:
            not_executable.append(path.as_posix())
    if not_executable:
        apply_rule(qs, rules, "script_not_executable", _listed(not_executable), ctx=ctx, file=not_executable[0])

    patterns = rules["tests_missing"].get("patterns", [])
    if not ctx.mods and not any(_matches_test_pattern(path, patterns) for path in ctx.source_files):
        apply_rule(qs, rules, "tests_missing", ctx=ctx)
    if ctx.mods and not mod_test_files(ctx.source_files):
        apply_rule(qs, rules, "mod_tests_missing", ctx=ctx, file=ctx.mods[0].hooks_file)

    if ctx.readme is not None:
        headings = rules["readme_limitations_missing"].get("headings", [])
        if not any(heading in ctx.readme for heading in headings):
            apply_rule(qs, rules, "readme_limitations_missing", ctx=ctx, file=_readme_file(ctx))


_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\s*\1\b", re.DOTALL)


def static_output_chars(command: str, ctx: PluginQualityContext, args: list[str] | None = None) -> int:
    """Largest fixed text a hook command prints: echo/printf literals, a heredoc, or ``cat`` of a plugin file.

    Dynamic output (anything with ``$`` or backticks, or a script) counts as 0.
    """
    if args is not None:  # exec form: no shell, so no heredoc
        sizes: list[int] = []
        words = [command, *args]
    else:
        sizes = [len(match.group(2)) for match in _HEREDOC_RE.finditer(command)]
        words = _split(command)
    if len(words) >= 2 and words[0] in {"echo", "printf"} and not any("$" in w or "`" in w for w in words[1:]):
        sizes.append(len(" ".join(word for word in words[1:] if not word.startswith("-"))))
    if len(words) == 2 and words[0] == "cat":
        for path, via_root, _interp in hook_command_scripts(words[1], ctx):
            if via_root:
                text = read_plugin_text(ctx, path)
                sizes.append(len(text) if text is not None else 0)
    return max(sizes, default=0)


def check_efficiency(ctx: PluginQualityContext, qs: PluginQualityScoreResult, rules: dict) -> None:
    cost = ctx.inventory.context_cost()
    qs.always_on_tokens = int(cost.get("always_on_tokens", 0))
    max_tokens = rules["context_cost_high"].get("max_tokens", 5000)
    if qs.always_on_tokens > max_tokens:
        top = sorted(cost.get("by_component", []), key=lambda row: row.get("always_on_tokens", 0), reverse=True)
        detail = f"~{qs.always_on_tokens} tokens > {max_tokens}"
        if top and top[0].get("always_on_tokens"):
            detail += f"; largest: {top[0]['type']} {top[0]['name']} (~{top[0]['always_on_tokens']})"
        apply_rule(qs, rules, "context_cost_high", detail, ctx=ctx)

    # Literal matcher check: matcher_scope fails closed for "all" and would count undecidable matchers too.
    broad_events = set(rules["hook_matcher_broad"].get("events", []))
    broad = [view for view in ctx.handlers if view.event in broad_events and view.matcher in _MATCH_ALL_MATCHERS]
    if broad:
        apply_rule(
            qs,
            rules,
            "hook_matcher_broad",
            _listed([f"{view.file}:{view.event}" for view in broad]),
            ctx=ctx,
            file=broad[0].file,
        )

    # Sized from the full handler command: HookRecord.target (which the context-cost
    # rows use) is truncated, so a long literal would be under-counted there.
    static_rule = rules["hook_static_context_large"]
    static_events = set(static_rule.get("events", []))
    max_chars = static_rule.get("max_chars", 2000)
    large: list[tuple[HookHandlerView, int]] = []
    for view in _command_handlers(ctx):
        if view.event not in static_events:
            continue
        args = view.handler.get("args")
        size = static_output_chars(
            view.handler["command"], ctx, [str(a) for a in args] if isinstance(args, list) else None
        )
        if size > max_chars:
            large.append((view, size))
    if large:
        apply_rule(
            qs,
            rules,
            "hook_static_context_large",
            _listed([f"{view.file}:{view.event} (~{size} chars)" for view, size in large]),
            ctx=ctx,
            file=large[0][0].file,
        )

    unfiltered_events = set(rules["mod_unfiltered_handler"].get("events", []))
    unfiltered = [h for mod in ctx.mods for h in mod.handlers if h.event in unfiltered_events and not h.filtered]
    if unfiltered:
        apply_rule(
            qs,
            rules,
            "mod_unfiltered_handler",
            _listed([f"{h.event} ({h.file}:{h.line})" for h in unfiltered]),
            ctx=ctx,
            file=unfiltered[0].file,
            line=unfiltered[0].line,
        )


def component_counts(inventory: PluginInventory) -> dict[str, Any]:
    counts = inventory.component_inventory().get("counts", {})
    return dict(counts) if isinstance(counts, dict) else {}
