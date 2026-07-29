#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Web4 Contributors
#
# Test suite for shell-tool lineage equivalence in the legacy mirror

"""
The legacy engine must judge a shell act the same whether the caller names the
tool "Bash" (Claude Code) or "Shell" (gemini-cli's run_shell_command, via the
lineage map). Sibling of dp-web4/hestia#107, which closed the same hole in the
Rust presets the live daemon enforces.

Two things shape these tests:

1. The rules are exercised through the hook's own classify_action /
   extract_target / full_command, never with hand-written category and target
   strings. Adding "Shell" to the four `tools=` lines is inert on its own —
   all three helpers were Bash-only, so a Shell act arrived category="other",
   target="", full_command=None and every rule still no-opped. A test that
   feeds the preset its inputs directly cannot see that, which is why the
   existing suite passes on both sides of this fix.

2. The hole being closed was a FALLTHROUGH-TO-ALLOW, so an equivalence
   assertion alone would be satisfied by both names going permissive — the bug,
   not the fix. `test_whitelist_rule_fires_under_both_names` pins a rule_id that
   can only be reached when the tool matched AND the act was classified AND the
   command line was passed through, under each name separately.

See test_destructive_deny_is_vacuous for a larger, pre-existing hole this fix
deliberately does NOT close.
"""

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from governance.policy_entity import PolicyRegistry

PLUGIN_DIR = Path(__file__).parent

# Deliberately a literal, not `from governance.presets import SHELL_TOOLS`.
# This is the specification the tests assert against; importing it from the
# code under test would make the assertions tautological, and — more to the
# point — would turn the negative control into an ImportError, which
# demonstrates nothing about behaviour.
SHELL_TOOLS = ["Bash", "Shell"]


def _load_hook():
    """Load hooks/pre_tool_use.py as a module.

    It is a script, not a package member, and its own imports (heartbeat,
    slot_channel) resolve relative to hooks/ — so that directory goes on the
    path before exec.
    """
    hooks_dir = PLUGIN_DIR / "hooks"
    sys.path.insert(0, str(hooks_dir))
    spec = importlib.util.spec_from_file_location(
        "web4_pre_tool_use", hooks_dir / "pre_tool_use.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


@pytest.fixture
def temp_storage():
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def safety(temp_storage):
    return PolicyRegistry(temp_storage).register_policy("safety", preset="safety")


def judge(entity, tool_name, command):
    """Evaluate a shell act exactly as the hook's main() does.

    main() takes category and target from the R6 request (create_r6_request ->
    classify_action / extract_target) and computes full_command itself. This
    reproduces that chain rather than short-circuiting it.

    full_command keys off the category rather than naming the hook's shell-tool
    set: classify_action maps exactly that set to "command", so the two are
    equivalent by construction, and this way the helper runs unmodified against
    a pre-fix hook — which is what makes the negative control meaningful.
    """
    tool_input = {"command": command}
    category = hook.classify_action(tool_name)
    target = hook.extract_target(tool_name, tool_input)
    full_command = tool_input.get("command") if category == "command" else None
    return entity.evaluate(tool_name, category, target, None, full_command)


# Spread across the arms of the safety preset: the whitelist allow, a plain
# fallthrough, and the two rules that turn out to be unreachable (below).
LINEAGE_CASES = [
    "rm -rf /tmp/scratch",       # allow-rm-whitelisted-scratch
    "ls -la /home/user",         # no rule -> default
    "rm -rf /home/user/data",    # deny-destructive-commands, were it reachable
    "rm /home/user/notes.txt",   # warn-file-delete, were it reachable
]


@pytest.mark.parametrize("command", LINEAGE_CASES)
def test_shell_judged_like_bash(safety, command):
    """Same command, both tool names, same verdict AND same rule.

    Equivalence over rule_id as well as decision: two names agreeing on "allow"
    for different reasons would still be a lineage defect.
    """
    as_bash = judge(safety, "Bash", command)
    as_shell = judge(safety, "Shell", command)

    assert as_shell.decision == as_bash.decision, (
        f"{command!r} judged {as_bash.decision} as Bash "
        f"but {as_shell.decision} as Shell"
    )
    assert as_shell.rule_id == as_bash.rule_id, (
        f"{command!r} matched rule {as_bash.rule_id} as Bash "
        f"but {as_shell.rule_id} as Shell"
    )


@pytest.mark.parametrize("tool_name", SHELL_TOOLS)
def test_whitelist_rule_fires_under_both_names(safety, tool_name):
    """A named rule is reached under each name — the fix, not just symmetry.

    `allow-rm-whitelisted-scratch` matches on command_patterns, so reaching it
    requires the tool to be in the rule's `tools`, the act to classify as a
    command, and the full command line to have been passed through. Any one of
    those regressing to Bash-only drops this to rule_id=None for "Shell".
    """
    result = judge(safety, tool_name, "rm -rf /tmp/scratch")
    assert result.decision == "allow"
    assert result.rule_id == "allow-rm-whitelisted-scratch"


@pytest.mark.parametrize("tool_name", SHELL_TOOLS)
def test_shell_tools_classify_as_commands(tool_name):
    """The three Bash-only helpers that made `tools=` inert.

    Guards the fix at its actual seam: if any regresses, the preset rules stop
    matching for "Shell" even though the rule lists it.
    """
    tool_input = {"command": "rm -rf /home/user/data"}
    # classify_action gates both the category rules and, transitively,
    # full_command; extract_target gates the target rules.
    assert hook.classify_action(tool_name) == "command"
    assert hook.extract_target(tool_name, tool_input) == "rm"


@pytest.mark.parametrize("tool_name", SHELL_TOOLS)
@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: extract_target truncates to the program name, so "
           "target_patterns can never match. Not a lineage defect — holds for "
           "Bash too. See the docstring.",
)
def test_destructive_deny_is_vacuous(safety, tool_name):
    """A hole strictly larger than the one this PR closes, pinned as xfail.

    `deny-destructive-commands` matches `target_patterns=[r"rm\\s+-"]` against
    `target`. In hestia's Rust engine that field carries the WHOLE command
    (presets.rs: "handler.rs hands the whole command in as target"). The mirror's
    extract_target returns `cmd.split()[0]` — "rm" — which contains no
    whitespace, so the pattern cannot match under ANY tool name. Same for
    warn-file-delete. Only the whitelist rule is live, because it keys on
    command_patterns, which do get the full line.

    test_policy_entity.py::test_evaluate_deny_destructive passes because it
    calls evaluate() with the full command as `target` directly. That is the
    input the production path never produces.

    Left open deliberately. The naive repair — hand extract_target's shell
    branch the whole command — makes the rule live but WITHOUT Rust's
    `target_patterns_scope: MatchScope::ExecutablePositions`, so it would deny
    `grep "rm -rf" log` for saying the word. That false-positive class was
    already paid for once on the Rust side (ten denies on one member,
    2026-07-27); reintroducing it here to close a gap that only affects
    fail-open members during a daemon outage is the wrong trade to make
    silently. Closing it properly means porting policy::shell, which is a
    decision about whether this mirror should exist, not a bug fix.
    """
    result = judge(safety, tool_name, "rm -rf /home/user/data")
    assert result.decision == "deny"
    assert result.rule_id == "deny-destructive-commands"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
