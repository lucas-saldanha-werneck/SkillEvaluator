# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static, read-only view of a plugin's mods (hooks modules) for the plugin quality score.

A mod is a Claude Code plugin whose ``hooks.json`` lists hooks modules under
``"modules"``. Nothing here runs plugin code: modules are located, scanned as
text, and only string literals are interpreted.

Checked against https://code.claude.com/docs/en/plugins/mods/reference,
/plugins/mods/events and /plugins/mods/interface on 2026-11-13:

(a) A hooks module "Exports ``register(on, options)``" as a named export; the
    docs show no ``export default`` form, so only a named ``register`` export
    is accepted. Module paths are relative to the hooks file. Accepted module
    suffixes: ``.js .mjs .cjs .jsx .ts .mts .cts .tsx``.
(b) Any event takes a filter ("matcher") object as the second argument to
    ``on``. ``ui.render`` included: "A ``ui.render`` hook runs for every render
    site unless you narrow it", so it is listed with ``tool.call`` in the
    packaged ``mod_unfiltered_handler`` rule.

Heuristics:

- ``$.tool.register({...})`` / ``$.command.register({...})``: the object literal
  is cut out with a scanner that respects quotes, template literals, comments
  and braces (up to ``PLUGIN_MOD_LITERAL_MAX_CHARS``), and its top-level
  ``name`` and ``description`` keys are read. A value counts only when it is a
  string literal (quotes, or backticks without ``${``) or literals joined with
  ``+``; anything else is ``resolved=False`` and is never scored.
- Handlers: the parameter name of ``register(<p>)`` (``on`` by default);
  ``<p>('event', {`` is filtered, ``<p>('event', <anything else>`` is not.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    PLUGIN_MOD_LITERAL_MAX_CHARS,
    PLUGIN_MOD_MAX_FILES,
    PLUGIN_MOD_SOURCE_SUFFIXES,
)
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot

if TYPE_CHECKING:
    from skillevaluator.plugin_discovery import PluginMember

_EXPORT_RES = (
    re.compile(r"\bexport\s+(?:async\s+)?function\s*\*?\s*register\b"),
    re.compile(r"\bexport\s+(?:const|let|var)\s+register\b"),
    re.compile(r"\bexport\s*\{[^}]*\bregister\b[^}]*\}"),
)
_REGISTER_PARAM_RES = (
    re.compile(r"\bfunction\s*\*?\s*register\s*\(\s*([A-Za-z_$][\w$]*)"),
    re.compile(
        r"\bregister\s*(?::[^=\n]+)?=\s*(?:async\s+)?(?:function\s*\*?\s*\w*\s*)?\(\s*([A-Za-z_$][\w$]*)"
    ),
    re.compile(r"\bregister\s*(?::[^=\n]+)?=\s*(?:async\s+)?([A-Za-z_$][\w$]*)\s*=>"),
)
_REGISTRATION_RE = re.compile(r"\$\s*\.\s*(tool|command)\s*\.\s*register\s*\(")


@dataclass(frozen=True)
class ModRegistration:
    """One ``$.tool.register`` / ``$.command.register`` call site."""

    kind: Literal["tool", "command"]
    name: str | None
    description: str | None
    resolved: bool
    file: str
    line: int


@dataclass(frozen=True)
class ModHandler:
    """One ``on('event', ...)`` call inside a hooks module."""

    event: str
    filtered: bool
    file: str
    line: int


@dataclass
class ModInventory:
    """What one hooks config with ``"modules"`` declares and what its source registers."""

    hooks_file: str
    modules: list[str] = field(default_factory=list)
    module_problems: list[tuple[str, str]] = field(default_factory=list)
    registrations: list[ModRegistration] = field(default_factory=list)
    handlers: list[ModHandler] = field(default_factory=list)
    files_scanned: int = 0
    truncated: bool = False


def _is_test_file(path: PurePosixPath) -> bool:
    return ".test." in path.name or ".spec." in path.name


def _in_node_modules(path: PurePosixPath) -> bool:
    return "node_modules" in path.parts


def mod_test_files(source_files: Iterable[PurePosixPath]) -> list[PurePosixPath]:
    """Mod tests (``*.test.ts`` / ``*.test.tsx``) outside ``node_modules``, as ``claude plugin test`` runs them."""
    return [
        path
        for path in source_files
        if path.name.endswith((".test.ts", ".test.tsx")) and not _in_node_modules(path)
    ]


def mask_comments(source: str) -> str:
    """Blank out ``//`` and ``/* */`` comments, keeping strings, offsets and newlines intact."""
    out = list(source)
    i, n = 0, len(source)
    quote: str | None = None
    while i < n:
        ch = source[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            i += 1
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            while i < n and source[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            for j in range(i, end):
                if out[j] != "\n":
                    out[j] = " "
            i = end
            continue
        i += 1
    return "".join(out)


def _skip_string(text: str, i: int) -> int:
    """Index just past the string literal that starts at ``text[i]`` (or ``len(text)``)."""
    quote = text[i]
    i += 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    return len(text)


def _object_literal(text: str, start: int) -> str | None:
    """The ``{...}`` literal starting at ``text[start]``, or ``None`` if unbalanced within the limit."""
    depth = 0
    i = start
    limit = min(len(text), start + PLUGIN_MOD_LITERAL_MAX_CHARS)
    while i < limit:
        ch = text[i]
        if ch in "'\"`":
            i = _skip_string(text, i)
            continue
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return None


def _top_level_entries(obj: str) -> dict[str, str]:
    """Top-level ``key: value`` source text of an object literal (``{`` and ``}`` included in ``obj``)."""
    entries: dict[str, str] = {}
    body = obj[1:-1]
    parts: list[str] = []
    depth, i, last = 0, 0, 0
    while i < len(body):
        ch = body[i]
        if ch in "'\"`":
            i = _skip_string(body, i)
            continue
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[last:i])
            last = i + 1
        i += 1
    parts.append(body[last:])
    for part in parts:
        match = re.match(r"\s*(?:([A-Za-z_$][\w$]*)|'([^']*)'|\"([^\"]*)\")\s*:", part)
        if match is None:
            continue
        key = next(group for group in match.groups() if group is not None)
        entries.setdefault(key, part[match.end() :].strip())
    return entries


def _literal_value(expression: str) -> str | None:
    """The string a literal (or literals joined with ``+``) evaluates to, else ``None``."""
    pieces: list[str] = []
    i, n = 0, len(expression)
    expect_literal = True
    while i < n:
        ch = expression[i]
        if ch.isspace():
            i += 1
            continue
        if expect_literal:
            if ch not in "'\"`":
                return None
            end = _skip_string(expression, i)
            raw = expression[i + 1 : end - 1]
            if end > n or expression[end - 1] != ch or (ch == "`" and "${" in raw):
                return None
            pieces.append(_unescape(raw))
            i = end
            expect_literal = False
        else:
            if ch != "+":
                return None
            i += 1
            expect_literal = True
    if expect_literal:
        return None
    return "".join(pieces)


def _unescape(raw: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), raw)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _registrations(text: str, rel: str) -> list[ModRegistration]:
    found: list[ModRegistration] = []
    for match in _REGISTRATION_RE.finditer(text):
        kind: Literal["tool", "command"] = "tool" if match.group(1) == "tool" else "command"
        line = _line_of(text, match.start())
        j = match.end()
        while j < len(text) and text[j].isspace():
            j += 1
        obj = _object_literal(text, j) if j < len(text) and text[j] == "{" else None
        if obj is None:
            found.append(ModRegistration(kind, None, None, False, rel, line))
            continue
        entries = _top_level_entries(obj)
        name = _literal_value(entries["name"]) if "name" in entries else None
        if "description" in entries:
            description = _literal_value(entries["description"])
            resolved = description is not None
        else:
            description, resolved = None, True  # no description key at all: known to be missing
        found.append(ModRegistration(kind, name, description, resolved, rel, line))
    return found


def _register_param(text: str) -> str:
    for pattern in _REGISTER_PARAM_RES:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return "on"


def _handlers(text: str, rel: str) -> list[ModHandler]:
    param = re.escape(_register_param(text))
    pattern = re.compile(rf"(?<![\w$.]){param}\s*\(\s*(['\"`])([^'\"`\n]+)\1\s*,\s*")
    return [
        ModHandler(match.group(2), text[match.end() : match.end() + 1] == "{", rel, _line_of(text, match.start()))
        for match in pattern.finditer(text)
    ]


def _exports_register(text: str) -> bool:
    return any(pattern.search(text) for pattern in _EXPORT_RES)


def _read_text(secure_root: SecureRoot, rel: PurePosixPath) -> str | None:
    try:
        raw, _metadata = secure_root.read_bytes(Path(rel), CONTENT_DEDUP_MAX_FILE_BYTES)
    except (SecurePathError, OSError, ValueError):
        return None
    return raw.decode("utf-8", errors="replace")


def _module_problems(
    raw_modules: object, base: str, files: set[PurePosixPath]
) -> tuple[list[str], list[tuple[str, str]], list[PurePosixPath]]:
    if isinstance(raw_modules, str):
        return [raw_modules], [(raw_modules, "not_list")], []
    if not isinstance(raw_modules, list):
        return [], [(str(raw_modules), "not_list")], []
    modules: list[str] = []
    problems: list[tuple[str, str]] = []
    resolved: list[PurePosixPath] = []
    for entry in raw_modules:
        if not isinstance(entry, str):
            problems.append((str(entry), "not_list"))
            continue
        modules.append(entry)
        joined = posixpath.normpath(posixpath.join(base, entry)) if not entry.startswith("/") else entry
        if joined.startswith("/") or joined == ".." or joined.startswith("../"):
            problems.append((entry, "outside_root"))
            continue
        path = PurePosixPath(joined)
        if path not in files:
            problems.append((entry, "missing"))
            continue
        resolved.append(path)
    return modules, problems, resolved


def read_mods(
    root: Path, hook_members: Iterable[PluginMember], source_files: Iterable[PurePosixPath]
) -> list[ModInventory]:
    """One :class:`ModInventory` per hooks config that has a ``"modules"`` key."""
    files = tuple(source_files)
    file_set = set(files)
    inventories: list[ModInventory] = []
    seen: set[str] = set()
    for member in hook_members:
        config = member.config
        if not isinstance(config, dict) or "modules" not in config or member.path is None:
            continue
        hooks_file = str(member.path)
        if hooks_file in seen:
            continue
        seen.add(hooks_file)
        base = posixpath.dirname(hooks_file)
        modules, problems, module_paths = _module_problems(config["modules"], base, file_set)
        inventory = ModInventory(hooks_file=hooks_file, modules=modules, module_problems=problems)
        base_path = PurePosixPath(base) if base else None
        candidates = [
            path
            for path in files
            if path.suffix in PLUGIN_MOD_SOURCE_SUFFIXES
            and not _is_test_file(path)
            and not _in_node_modules(path)
            and (base_path is None or base_path in path.parents)
        ]
        ordered = module_paths + [path for path in candidates if path not in module_paths]
        if len(ordered) > PLUGIN_MOD_MAX_FILES:
            inventory.truncated = True
            ordered = ordered[:PLUGIN_MOD_MAX_FILES]
        try:
            with SecureRoot(root) as secure_root:
                for path in ordered:
                    text = _read_text(secure_root, path)
                    if text is None:
                        continue
                    inventory.files_scanned += 1
                    masked = mask_comments(text)
                    if path in module_paths and not _exports_register(masked):
                        raw = next(m for m in modules if posixpath.normpath(posixpath.join(base, m)) == str(path))
                        inventory.module_problems.append((raw, "no_register_export"))
                    inventory.registrations.extend(_registrations(masked, path.as_posix()))
                    if path in module_paths:
                        inventory.handlers.extend(_handlers(masked, path.as_posix()))
        except (SecurePathError, OSError):
            inventory.truncated = True
        inventories.append(inventory)
    return inventories
