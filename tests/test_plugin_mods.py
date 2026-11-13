# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static reading of mod hooks modules: modules, registrations, handlers and tests."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from tests.validators.plugin_quality_support import GOOD_MANIFEST, make_plugin

from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.plugin_mods import mod_test_files, read_mods


def _mod(tmp_path, register_src, hooks=None):
    return make_plugin(
        tmp_path / "m",
        manifest=GOOD_MANIFEST,
        files={
            "hooks/hooks.json": json.dumps(hooks or {"modules": ["./register.ts"]}),
            "hooks/register.ts": register_src,
        },
    )


def _read(root):
    inv = plugin_inventory_for_root(root)
    return read_mods(inv.root, inv.of_type("hook"), inv.source_files)


def test_literal_tool_and_command_descriptions_are_read(tmp_path):
    root = _mod(
        tmp_path,
        "export function register(on) {\n on('session.start', async ($, e, next) => {\n"
        "  await $.tool.register({ name: 'ticket', description: 'Look up a ticket by id and return its status' })\n"
        "  await $.command.register({ name: 'standup', description: 'Sum' })\n  return next(e) }) }\n",
    )
    [mod] = _read(root)
    assert mod.module_problems == []
    assert [(r.kind, r.name, r.resolved) for r in mod.registrations] == [
        ("tool", "ticket", True),
        ("command", "standup", True),
    ]
    assert mod.registrations[1].description == "Sum" and mod.registrations[0].line == 3


def test_non_literal_description_is_unresolved(tmp_path):
    root = _mod(
        tmp_path,
        "const D = 'x'\nexport function register(on) { on('session.start', async ($) => {\n"
        " await $.tool.register({ name: 't', description: D })\n"
        " await $.tool.register({ name: 'u', description: `a ${D}` }) }) }\n",
    )
    [mod] = _read(root)
    assert [r.resolved for r in mod.registrations] == [False, False]
    assert [r.description for r in mod.registrations] == [None, None]


def test_concatenated_and_nested_literals(tmp_path):
    root = _mod(
        tmp_path,
        "export function register(on) { on('session.start', async ($) => {\n"
        " await $.tool.register({ name: \"t\", input: { description: 'inner' },\n"
        "   description: 'Reads the ' + \"ticket, then \" + `returns it`, run: async () => ({}) }) }) }\n",
    )
    [mod] = _read(root)
    [reg] = mod.registrations
    assert reg.resolved and reg.description == "Reads the ticket, then returns it"


def test_commented_out_registration_is_ignored(tmp_path):
    root = _mod(
        tmp_path,
        "export function register(on) {\n // $.tool.register({ name: 'x', description: 'y' })\n"
        " /* $.command.register({ name: 'z' }) */ }\n",
    )
    [mod] = _read(root)
    assert mod.registrations == []


def test_missing_module_and_missing_export_are_problems(tmp_path):
    root = _mod(tmp_path, "export const other = 1\n", hooks={"modules": ["./register.ts", "./gone.ts"]})
    [mod] = _read(root)
    assert sorted(mod.module_problems) == [("./gone.ts", "missing"), ("./register.ts", "no_register_export")]


def test_module_outside_root_is_problem(tmp_path):
    root = _mod(tmp_path, "export function register(on) {}\n", hooks={"modules": ["../../escape.ts"]})
    [mod] = _read(root)
    assert mod.module_problems == [("../../escape.ts", "outside_root")]


def test_reexported_register_is_accepted(tmp_path):
    root = _mod(tmp_path, "function register(on) {}\nexport { register }\n")
    [mod] = _read(root)
    assert mod.module_problems == []


def test_modules_not_a_list_is_problem(tmp_path):
    [mod] = _read(_mod(tmp_path, "export function register(on) {}\n", hooks={"modules": "./register.ts"}))
    assert mod.module_problems == [("./register.ts", "not_list")]


def test_hooks_without_modules_is_not_a_mod(tmp_path):
    root = make_plugin(
        tmp_path / "h", manifest=GOOD_MANIFEST, files={"hooks/hooks.json": json.dumps({"hooks": {"Stop": []}})}
    )
    assert _read(root) == []


def test_unfiltered_and_filtered_handlers(tmp_path):
    root = _mod(
        tmp_path,
        "export function register(hook) {\n hook('tool.call', async ($, e, next) => next(e))\n"
        " hook('command.run', { command: 'x' }, async () => ({}))\n}\n",
    )
    [mod] = _read(root)
    assert [(h.event, h.filtered) for h in mod.handlers] == [("tool.call", False), ("command.run", True)]
    assert mod.handlers[0].line == 2 and mod.handlers[0].file == "hooks/register.ts"


def test_mod_test_files_found(tmp_path):
    root = _mod(tmp_path, "export function register(on) {}\n")
    (root / "tests").mkdir()
    (root / "tests" / "a.test.ts").write_text("test('x', () => {})", encoding="utf-8")
    inv = plugin_inventory_for_root(root)
    assert mod_test_files(inv.source_files) == [PurePosixPath("tests/a.test.ts")]
