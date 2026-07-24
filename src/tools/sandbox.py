"""Sandboxed execution of LLM-generated pandas/sklearn code.

Two gates, in order:

1. `validate_code` — static AST check. Code that fails never runs, so a blocked
   import costs nothing and the error message is precise.
2. `_sandbox_child` — a separate `python -I` process with an import guard, an
   `open()` path guard, an address-space cap and a wall-clock kill.

Nothing here is a substitute for the process boundary: the child is the only
thing standing between generated code and this machine, so the parent never
execs generated code itself, not even to "check" it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .sandbox_policy import ALLOWED_IMPORTS, BANNED_IMPORTS, validate_code

_CHILD = Path(__file__).with_name("_sandbox_child.py")


@dataclass
class ExecResult:
    """Outcome of one sandboxed execution."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    result: object = None
    traceback: str = ""
    violations: list[str] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    timed_out: bool = False

    @property
    def error_feedback(self) -> str:
        """What to hand back to an agent so it can correct its own code."""
        if self.violations:
            return "The code was rejected before execution:\n" + "\n".join(
                f"- {v}" for v in self.violations
            )
        if self.timed_out:
            return "The code exceeded the execution timeout and was killed."
        return self.traceback or self.stderr


def _read_roots() -> list[str]:
    """Paths generated code may read from: the stdlib and installed packages.

    sklearn and plotly read bundled data files at import time, so a read guard
    that only allowed the run directory would break them.
    """
    roots = {sysconfig.get_paths()[k] for k in ("stdlib", "purelib", "platlib")}
    roots.add(str(Path(sys.executable).parent))
    return sorted(r for r in roots if r)


def run_code(
    code: str,
    run_dir: Path,
    timeout_s: int = 120,
    memory_mb: int = 2048,
) -> ExecResult:
    """Validate, then execute generated code with the run directory as its cwd.

    Artifacts are detected by diffing the run directory, so a plot the code
    writes is picked up without the code having to declare it.

    Note: the memory cap uses POSIX rlimits and is a no-op on Windows, where only
    the wall-clock timeout applies. Deployment is Linux via docker-compose, so
    the cap holds in production and degrades on a Windows dev box.
    """
    violations = validate_code(code)
    if violations:
        return ExecResult(ok=False, violations=violations)

    run_dir = Path(run_dir)
    work = run_dir / "_sandbox"
    work.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex[:8]
    code_path = work / f"code_{token}.py"
    job_path = work / f"job_{token}.json"
    result_path = work / f"result_{token}.json"
    code_path.write_text(code, encoding="utf-8")

    job_path.write_text(
        json.dumps(
            {
                "code_path": str(code_path.resolve()),
                "run_dir": str(run_dir.resolve()),
                "result_path": str(result_path.resolve()),
                "memory_mb": memory_mb,
                "timeout_s": timeout_s,
                "allowed": sorted(ALLOWED_IMPORTS),
                "banned": sorted(BANNED_IMPORTS),
                "read_roots": _read_roots(),
            }
        ),
        encoding="utf-8",
    )

    before = _snapshot(run_dir)

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(_CHILD), str(job_path)],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            ok=False,
            timed_out=True,
            traceback=f"Execution exceeded {timeout_s}s and was terminated.",
            artifacts=_new_artifacts(run_dir, before),
        )

    artifacts = _new_artifacts(run_dir, before)

    if not result_path.exists():
        # The child died before it could write a result — OOM kill, or a crash
        # in the runner itself. Its stderr is all we have.
        return ExecResult(
            ok=False,
            stdout=proc.stdout,
            stderr=proc.stderr,
            traceback=proc.stderr or "Sandbox process exited without a result.",
            artifacts=artifacts,
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return ExecResult(
        ok=bool(payload["ok"]),
        stdout=payload.get("stdout", ""),
        stderr=payload.get("stderr", ""),
        result=payload.get("result"),
        traceback=payload.get("traceback", ""),
        artifacts=artifacts,
    )


def _snapshot(run_dir: Path) -> set[Path]:
    return {p for p in run_dir.rglob("*") if p.is_file()}


def _new_artifacts(run_dir: Path, before: set[Path]) -> list[Path]:
    """Files the execution created, excluding the sandbox's own scratch files."""
    work = run_dir / "_sandbox"
    return sorted(
        p for p in _snapshot(run_dir) - before if work not in p.parents and p != work
    )
