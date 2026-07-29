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

See test_destructive_deny_rm_limb_is_dead for a larger, pre-existing hole this
fix deliberately does NOT close.
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
# fallthrough, the one destructive limb that actually fires, and the two that
# turn out to be unreachable (below).
LINEAGE_CASES = [
    "rm -rf /tmp/scratch",       # allow-rm-whitelisted-scratch
    "ls -la /home/user",         # no rule -> default
    "mkfs.ext4 /dev/sdb1",       # deny-destructive-commands, mkfs limb: LIVE
    "rm -rf /home/user/data",    # deny-destructive-commands, rm limb: dead
    "rm /home/user/notes.txt",   # warn-file-delete, dead
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
    reason="pre-existing: extract_target truncates to the first token, which "
           "cannot contain whitespace, so the `rm\\s+-` limb can never match. "
           "Not a lineage defect — holds for Bash too. NOTE: the sibling "
           "`mkfs\\.` limb of the same rule DOES fire; see the docstring "
           "before concluding this pin is stale.",
)
def test_destructive_deny_rm_limb_is_dead(safety, tool_name):
    """A hole strictly larger than the one this PR closes, pinned as xfail.

    THE RULE IS NOT VACUOUS — one of its two limbs is. Read this before
    "fixing" or removing the pin.

    A shell act's `target` is `cmd.split()[0]`, which is whitespace-free by
    construction. So, over the whole preset:

        a target_pattern is LIVE iff some whitespace-free string matches it.

    Applying that to the shell-scoped rules (verified empirically through the
    real classify_action -> extract_target -> evaluate seam, both tool names):

        rule                          pattern        verdict
        deny-destructive-commands     'rm\\s+-'       DEAD  — needs whitespace
        deny-destructive-commands     'mkfs\\.'       LIVE  — prefix of 'mkfs.ext4'
        warn-file-delete              'rm\\s+[^-]'    DEAD  — needs whitespace
        allow-rm-whitelisted-scratch  (none)         LIVE  — keys on command_patterns
        warn-git-push-no-pat          (none)         LIVE  — keys on command_patterns

    "LIVE" for `mkfs\\.` is the weakest of the three: the pattern needs the
    LITERAL DOT, so the portable `mkfs -t ext4 /dev/sdb1` is dead with no
    wrapper at all. That is a second and independent narrowing, and unlike the
    first-token one it survives the repair below — see
    test_destructive_deny_mkfs_limb_requires_literal_dot.

    In hestia's Rust engine `target` carries the WHOLE command (presets.rs:
    "handler.rs hands the whole command in as target"), scoped by
    `target_patterns_scope: MatchScope::ExecutablePositions`. The rules were
    lifted verbatim across that difference. test_policy_entity.py::
    test_evaluate_deny_destructive passes because it calls evaluate() with the
    full command as `target` directly — an input the production path never
    produces.

    The live limb is worth less than it looks, and in the same way: it matches
    only when `mkfs.*` is the literal first token, so `sudo mkfs.ext4 /dev/sdb1`
    — the only form in which mkfs actually runs — allows, as do `time`, `env`,
    `nohup` and anything chained. See test_destructive_deny_mkfs_limb_is_first
    _token_only. So the two failures are one defect seen from two sides:
    `target` is neither the whole command nor the executable positions, it is
    the first token, which is wrong in both directions at once.

    Left open deliberately. The naive fix — hand extract_target's shell branch
    the whole command — was measured rather than assumed, and it costs only
    precision, never coverage:

        case                        today (first token)  naive (whole command)
        mkfs.ext4 /dev/sdb1         deny                 deny
        sudo mkfs.ext4 /dev/sdb1    allow                deny      (gained)
        rm -rf /home/user/data      allow                deny      (gained)
        grep "mkfs.ext4" syslog     allow                deny      <-- FALSE POS
        grep "rm -rf" syslog        allow                deny      <-- FALSE POS
        rm -rf /tmp/scratch         allow-whitelist      allow-whitelist
        mkfs -t ext4 /dev/sdb1      allow                allow     <-- STILL DEAD
        sudo mkfs -t ext4 /dev/sdb1 allow                allow     <-- STILL DEAD

    The last two rows are the ones that tell the porter widening `target` is
    not the whole job: the PATTERN is also wrong. `mkfs\\.` cannot match an
    un-dotted invocation at any scope, so a real, unwrapped, root-level
    destructive command stays uncovered on both sides of the repair.
    (Cross-checked independently on two rigs, 2026-07-28.)

    So it fixes both limbs and breaks neither; what it reintroduces is
    deny-the-mention, because it flips `mkfs\\.` and `rm\\s+-` from token-scoped
    to match-anywhere with no `MatchScope::ExecutablePositions` equivalent.
    That false-positive class was already paid for once on the Rust side (ten
    denies on one member, 2026-07-27) — which is the whole reason the Rust rule
    carries that scope and this one has nowhere to put it.

    Worth noting for whoever does the port: the /tmp whitelist survives the
    widening on priority, so the escape hatch is not collateral damage.

    Closing this properly means porting policy::shell — a decision about
    whether this mirror should exist, not a bug fix.
    """
    result = judge(safety, tool_name, "rm -rf /home/user/data")
    assert result.decision == "deny"
    assert result.rule_id == "deny-destructive-commands"


@pytest.mark.parametrize("tool_name", SHELL_TOOLS)
def test_destructive_deny_mkfs_limb_fires_unwrapped(safety, tool_name):
    """The limb that IS live, pinned as the baseline for any repair.

    Deliberately not an xfail: it passes today and must keep passing. It does
    NOT guard against widening `target` to the whole command — that was
    measured and leaves this case denying (see the table above). What it
    catches is the opposite mistake: a repair that anchors or narrows the
    pattern and drops the one true positive the mirror currently gets.
    """
    result = judge(safety, tool_name, "mkfs.ext4 /dev/sdb1")
    assert result.decision == "deny"
    assert result.rule_id == "deny-destructive-commands"


# Every wrapper moves the real program out of cmd.split()[0]. `sudo` is the
# one that matters — mkfs needs root, so this is the realistic invocation.
WRAPPED_MKFS = [
    "sudo mkfs.ext4 /dev/sdb1",
    "time mkfs.ext4 /dev/sdb1",
    "env LC_ALL=C mkfs.ext4 /dev/sdb1",
    "nohup mkfs.ext4 /dev/sdb1",
    "true && mkfs.ext4 /dev/sdb1",
]


@pytest.mark.parametrize("command", WRAPPED_MKFS)
@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: target is cmd.split()[0], so any wrapper word hides "
           "the program from the only live destructive limb. Same root cause "
           "as test_destructive_deny_rm_limb_is_dead.",
)
def test_destructive_deny_mkfs_limb_is_first_token_only(safety, command):
    """The live limb's reach, measured rather than assumed.

    `/sbin/mkfs.ext4 ...` still denies (the pattern is unanchored, so it
    matches inside the path token). Anything that puts a DIFFERENT word first
    does not. Pinned because "the rule fires" is the wrong summary to leave
    behind for whoever decides the mirror's fate.
    """
    result = judge(safety, "Bash", command)
    assert result.decision == "deny"
    assert result.rule_id == "deny-destructive-commands"


# The dot is load-bearing. `mkfs -t ext4 /dev/sdb1` is the portable form and
# needs no wrapper to evade the rule — the first token IS the program, and it
# still does not match `mkfs\.`.
UNDOTTED_MKFS = [
    "mkfs -t ext4 /dev/sdb1",
    "sudo mkfs -t ext4 /dev/sdb1",
]


@pytest.mark.parametrize("command", UNDOTTED_MKFS)
@pytest.mark.xfail(
    strict=True,
    reason="pre-existing and DISTINCT from the first-token hole: the pattern "
           "'mkfs\\\\.' requires a literal dot, so the un-dotted invocation is "
           "uncovered even when the program is the first token — and unlike "
           "the wrapper cases, widening `target` does not fix it.",
)
def test_destructive_deny_mkfs_limb_requires_literal_dot(safety, command):
    """The second, independent narrowing of the one live destructive limb.

    Kept separate from test_destructive_deny_mkfs_limb_is_first_token_only on
    purpose. That test's subject is `target` scoping and every one of its cases
    flips to deny under the naive repair; these two do not flip, because their
    cause is the pattern rather than the scope. Folding them into the same
    parametrize would make that xfail's stated reason false for two of its
    rows, and would hide the one result that constrains the port: fixing scope
    alone still leaves a real destructive invocation allowed.
    """
    result = judge(safety, "Bash", command)
    assert result.decision == "deny"
    assert result.rule_id == "deny-destructive-commands"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
