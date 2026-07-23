# Security

## Threat model

This system **executes LLM-generated Python**. That is the entire security story:
a language model, steerable by whatever data lands in the uploaded CSV or the task
prompt, writes code that then runs on the host. Everything below exists to make
that safe enough to demo. The trust boundary is the generated code — it is treated
as hostile, because prompt injection through an uploaded dataset can make it
hostile without the operator noticing.

Assumed trusted: the operator, the source in this repo, the model provider.
Assumed hostile: every line of code the model emits, and every byte of an
uploaded dataset.

## The sandbox

Generated code runs in a separate `python -I -B` subprocess (`src/tools/sandbox.py`,
`src/tools/_sandbox_child.py`) under a two-layer policy (`src/tools/sandbox_policy.py`):

1. **Static gate (primary).** The code is `ast.parse`d and rejected *before
   execution* if it imports anything off a 15-module allow-list, imports anything
   on the banned list, uses a banned builtin, or accesses a dunder attribute.
2. **Runtime guards (defence in depth).** In the child: an `__import__` hook that
   re-checks imports from the generated frame, an `open()` hook confining writes to
   the run directory and reads to stdlib/site roots, POSIX rlimits on address space
   and CPU, and a parent-side wall-clock timeout.

### A real escape that was found and fixed

The static gate blocked dunder access written as an **attribute** (`x.__class__`)
but not as a **string** (`getattr(x, "__class__")`), because `getattr` was not a
banned name. That gap was a full sandbox escape, **confirmed by execution** against
the policy before the fix:

```python
# reached the real os module and ran os.listdir — no import, no dunder attribute
for c in getattr(getattr(getattr(0,"__class__"),"__base__"),"__subclasses__")():
    g = getattr(getattr(c,"__init__",None),"__globals__",None)
    if isinstance(g,dict) and "sys" in g:
        os = g["sys"].modules["os"]   # os is already loaded transitively
```

It walks `object.__subclasses__()` to any class whose `__globals__` holds a
reference to `sys`, then reads `os` out of `sys.modules` — so it never calls
`__import__` (which the runtime guard covers) and never writes a dunder attribute
(which the AST guard covers). From `os` it reaches `os.system` and file operations
that bypass the `open()` guard entirely.

**Fix:** `getattr`, `setattr`, `delattr` are now banned names
(`src/tools/sandbox_policy.py`). With dunder-attribute access already blocked by
the AST rule, there is no remaining route to obtain `__class__` / `__subclasses__`
/ `__globals__`, so the walk cannot start. Regression tests in
`tests/test_sandbox.py` (`test_getattr_sys_modules_escape_is_blocked` and the
parametrized escape cases) run the exact payload and assert it never executes.

## What is mitigated

| Risk | Status | Where |
|---|---|---|
| Arbitrary import in generated code | **Mitigated** — 15-module allow-list, static + runtime | `sandbox_policy.py`, `_sandbox_child.py:58` |
| `exec`/`eval`/`compile`/`__import__` in generated code | **Mitigated** — banned names | `sandbox_policy.py:38` |
| Dunder-attribute sandbox escape | **Mitigated** — AST rejects dunder attributes | `sandbox_policy.py:143` |
| **String-form dunder escape via `getattr`** | **Fixed** — `getattr`/`setattr`/`delattr` banned; confirmed by regression test | `sandbox_policy.py:38` |
| File write outside the run directory | **Mitigated** — `open()` write-guard | `_sandbox_child.py:79` |
| File read of arbitrary host paths | **Mitigated** — `open()` read-guard to stdlib/site + run dir | `_sandbox_child.py:79` |
| Runaway CPU / memory | **Mitigated on Linux** — rlimits + wall-clock timeout | `sandbox.py`, `_sandbox_child.py:24` |
| Runaway LLM spend | **Mitigated** — per-run USD cap, all traffic through one chokepoint | `src/tools/llm.py` |
| Container running as root | **Mitigated** — both images run as uid 1000 | `Dockerfile:30`, `dashboard/Dockerfile:23` |
| Dependency CVEs | **Clean** — `pip-audit`: no known vulnerabilities |
| Secrets in git history | **Clean** — `gitleaks`: 0 findings; no `.env` ever tracked |

## What is NOT mitigated

- **The memory/CPU rlimits are POSIX-only.** On a Windows dev box only the
  wall-clock timeout applies; the address-space cap is a no-op. Deployment is Linux
  (docker-compose), where it holds. This is stated at `sandbox.py:77`.
- **No authentication on the API.** It is a single-operator tool.
- **Prompt injection through uploaded data.** A crafted CSV can steer what code the
  model writes. The sandbox is the mitigation — injection can make the model *try*
  something hostile, but the policy is what stops it landing. There is no attempt to
  stop the injection itself.
- **The allow-list is a denylist's optimist twin.** It is small and audited, but a
  future escape through an *allowed* library (a pandas/numpy code path that reaches
  the filesystem) is not structurally impossible. New capability in those libraries
  should be reviewed against this policy.

## Reporting

Open an issue. Portfolio/demo project, no production deployment, no security SLA.
