"""Generate code with an LLM, run it in the sandbox, let it fix its own mistakes.

The retry loop is the whole point: a model that writes `df.groupby(...)` against a
column that doesn't exist gets the traceback back and usually fixes it on the
second attempt. Policy violations are fed back the same way, which turns "you
tried to import os" into a correction rather than a dead run.
"""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings
from ..state.schema import AgentMessage, RunState
from .llm import call_llm
from .sandbox import ExecResult, run_code

MAX_ATTEMPTS = 3

_SYSTEM = """You are a senior data scientist writing Python for a sandboxed runner.

Hard rules:
- You may import only: pandas, numpy, sklearn, plotly, optuna, math, json,
  statistics, datetime, re, collections, itertools, functools, typing, warnings.
- No os, sys, subprocess, pathlib, requests, pickle, eval, exec, or dunder access.
- The working directory is the run directory. Write files with relative paths only.
- Assign your answer to a variable named `result` holding JSON-safe data
  (dicts, lists, numbers, strings). Print nothing you don't need.
- Return ONLY the code. No markdown fences, no commentary.
"""


def _strip_fences(text: str) -> str:
    """Models add ```python fences even when told not to."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def generate_and_run(
    state: RunState,
    node: str,
    task: str,
    context: str = "",
) -> tuple[ExecResult, str, list[AgentMessage]]:
    """Write code for `task`, run it, and retry on failure with the error fed back.

    Returns the final result, the last code attempted, and log messages describing
    each attempt — including the failures, since "it got there on attempt 3" is
    something a reviewer should be able to see.
    """
    settings = get_settings()
    run_dir = settings.run_dir(state["run_id"])
    messages: list[AgentMessage] = []

    prompt = f"{context}\n\nTask:\n{task}" if context else f"Task:\n{task}"
    code = ""
    result = ExecResult(ok=False, traceback="No attempt was made.")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        code = _strip_fences(call_llm(state, node, _SYSTEM, prompt))
        result = run_code(
            code,
            run_dir=run_dir,
            timeout_s=settings.sandbox_timeout_s,
            memory_mb=settings.sandbox_memory_mb,
        )
        if result.ok:
            if attempt > 1:
                messages.append(
                    AgentMessage(
                        agent=node,
                        content=f"Code succeeded on attempt {attempt}.",
                    )
                )
            return result, code, messages

        messages.append(
            AgentMessage(
                agent=node,
                level="warning",
                content=(
                    f"Attempt {attempt}/{MAX_ATTEMPTS} failed: "
                    f"{result.error_feedback.strip().splitlines()[-1][:200]}"
                ),
            )
        )
        prompt = (
            f"{context}\n\nTask:\n{task}\n\n"
            f"Your previous attempt failed:\n```python\n{code}\n```\n\n"
            f"Error:\n{result.error_feedback}\n\n"
            "Fix the problem and return the corrected code."
        )

    messages.append(
        AgentMessage(
            agent=node,
            level="error",
            content=f"Code generation failed after {MAX_ATTEMPTS} attempts.",
        )
    )
    return result, code, messages


def run_dir_for(state: RunState) -> Path:
    """The artifact directory for this run."""
    return get_settings().run_dir(state["run_id"])
