#!/usr/bin/env python
"""Regression check for the prove-verifier's fail-closed behaviour.

`LeanServer.run()` is typed `CommandResponse | LeanError`, and `LeanError`
carries a single `message` field. Reading messages as
`getattr(response, "messages", [])` therefore turned a failed REPL call into a
clean compile: no errors, no `sorry`, `complete=True`.

The REPL reaches that path whenever an exception escapes `IO.processInput`
(missing module, failed `.olean` load, heartbeat interruption, stack overflow):
`REPL/Main.lean` turns it into `{"message": ...}` instead of messages carrying
`severity="error"`.

This script needs no Lean toolchain and no `[lean]` extra — `lean_interact` is
stubbed in `sys.modules`, so `verify()` runs its real code path.

    PYTHONPATH=src ./.venv/bin/python scripts/check_verifier.py
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --------------------------------------------------------------------------
# Stub `lean_interact` so the verifier can be exercised without the extra.
# --------------------------------------------------------------------------
class LeanError:
    """Mirrors the upstream model: a single `message` field, no `messages`."""

    def __init__(self, message: str = "") -> None:
        self.message = message

    def model_dump(self) -> dict:
        return {"message": self.message}


class _Pos:
    def __init__(self, line: int) -> None:
        self.line = line


class _Message:
    def __init__(self, severity: str, data: str, line: int = 0) -> None:
        self.severity = severity
        self.data = data
        self.start_pos = _Pos(line) if line else None

    def model_dump(self) -> dict:
        return {"severity": self.severity, "data": self.data}


class CommandResponse:
    """Mirrors the upstream model: `messages`, `sorries`, `env`."""

    def __init__(self, messages=(), sorries=()) -> None:
        self.messages = list(messages)
        self.sorries = list(sorries)
        self.env = 0

    def model_dump(self) -> dict:
        return {"messages": len(self.messages), "sorries": len(self.sorries)}


class Command:
    def __init__(self, cmd: str) -> None:
        assert cmd, "upstream Command forbids an empty cmd"
        self.cmd = cmd


def _install_stub() -> None:
    mod = types.ModuleType("lean_interact")
    mod.LeanError = LeanError  # type: ignore[attr-defined]
    mod.Command = Command  # type: ignore[attr-defined]
    mod.CommandResponse = CommandResponse  # type: ignore[attr-defined]
    sys.modules["lean_interact"] = mod


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
class _FakeServer:
    def __init__(self, response: object) -> None:
        self._response = response

    def run(self, _cmd: object, timeout: int = 0) -> object:  # noqa: ARG002
        return self._response


async def _verify_with(response: object):
    from leansearchv2.prove.verifier import LeanInteractVerifier

    verifier = LeanInteractVerifier(project_dir="/nonexistent")
    verifier._server = _FakeServer(response)  # bypass project setup
    try:
        return await verifier.verify("theorem foo : True := by trivial")
    finally:
        verifier._server = None


CASES: list[tuple[str, object, bool]] = [
    (
        "LeanError with a message (REPL-level exception)",
        LeanError("unknown module prefix 'Mathlib.GroupTheory.DoubleCoset'"),
        False,
    ),
    (
        "LeanError with an empty message (REPL returned {})",
        LeanError(""),
        False,
    ),
    (
        "clean CommandResponse (no errors, no sorry)",
        CommandResponse(messages=[_Message("info", "ok")]),
        True,
    ),
    (
        "CommandResponse with a compile error",
        CommandResponse(messages=[_Message("error", "unknown identifier 'foo'", line=12)]),
        False,
    ),
    (
        "CommandResponse with `sorry`",
        CommandResponse(messages=[_Message("warning", "declaration uses 'sorry'")], sorries=["x"]),
        False,
    ),
    (
        "response object lacking `messages` entirely",
        object(),
        False,
    ),
]


async def main() -> int:
    _install_stub()
    failures = 0
    for label, response, expect_success in CASES:
        result = await _verify_with(response)
        ok = result.success is expect_success
        failures += not ok
        mark = "ok  " if ok else "FAIL"
        detail = f"success={result.success}"
        if not expect_success and result.error_msg:
            detail += f" | {result.error_msg[:72]}"
        print(f"  [{mark}] {label}\n         {detail}")

        # The original bug: a LeanError yielded an empty message list, so the
        # old code path computed complete=True.
        if isinstance(response, LeanError):
            leaked = getattr(response, "messages", []) or []
            assert leaked == [], "stub drifted from upstream shape"
            print(f"         (旧逻辑会判 success={not leaked} <- 已修复)")

    # Error line numbers must survive into the reflection prompt.
    result = await _verify_with(
        CommandResponse(messages=[_Message("error", "unknown identifier 'foo'", line=12)])
    )
    if "line 12" not in result.error_msg:
        print(f"  [FAIL] error message lost its line number: {result.error_msg!r}")
        failures += 1
    else:
        print(f"  [ok  ] error message keeps its line number: {result.error_msg!r}")

    print()
    if failures:
        print(f"  {failures} case(s) FAILED")
        return 1
    print(f"  all {len(CASES) + 1} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
