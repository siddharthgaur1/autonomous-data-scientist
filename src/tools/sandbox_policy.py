"""The security policy for generated code. Imported by both parent and child.

Kept in its own module with no third-party imports so the sandbox child can load
it cheaply, and so the rules are readable in one place rather than scattered
across the runner.
"""

from __future__ import annotations

import ast

#: Top-level modules generated code may import. Anything else is rejected before
#: execution. Nested imports made *by* these libraries are not subject to the
#: list — only imports written in the generated code itself are.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "pandas",
        "numpy",
        "sklearn",
        "plotly",
        "optuna",
        "math",
        "json",
        "statistics",
        "datetime",
        "re",
        "collections",
        "itertools",
        "functools",
        "typing",
        "warnings",
    }
)

#: Names that are never callable from generated code, even though some are
#: builtins. `open` is absent from this list on purpose: it is wrapped by a path
#: guard at runtime rather than removed, since writing artifacts is legitimate.
BANNED_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "input",
        "memoryview",
        "exit",
        "quit",
    }
)

#: Rejected outright as imports. Redundant with the whitelist, but named
#: explicitly so a violation message says *why* rather than "not allowed".
BANNED_IMPORTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "urllib2",
        "urllib3",
        "requests",
        "httpx",
        "http",
        "ftplib",
        "telnetlib",
        "smtplib",
        "asyncio",
        "multiprocessing",
        "threading",
        "ctypes",
        "cffi",
        "pickle",
        "dill",
        "marshal",
        "shelve",
        "importlib",
        "pty",
        "signal",
        "resource",
        "pathlib",
        "glob",
        "tempfile",
        "webbrowser",
        "builtins",
        "gc",
        "inspect",
        "code",
        "codeop",
        "runpy",
        "atexit",
        "platform",
        "getpass",
        "pwd",
        "site",
    }
)


class PolicyViolation(Exception):
    """Generated code broke a static rule and was never executed."""


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def validate_code(code: str) -> list[str]:
    """Statically check generated code against the policy.

    Returns a list of human-readable violations; empty means the code is
    allowed to run. This is the primary gate — the runtime guards in the child
    are defence in depth, not the first line.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]

    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_check_import(_root(alias.name), alias.name))

        elif isinstance(node, ast.ImportFrom):
            # `from . import x` has no module; relative imports are meaningless here.
            if node.level and not node.module:
                violations.append("Relative imports are not allowed.")
                continue
            if node.module:
                violations.extend(_check_import(_root(node.module), node.module))

        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            violations.append(f"Use of '{node.id}' is not allowed.")

        elif isinstance(node, ast.Attribute) and _is_dunder(node.attr):
            violations.append(
                f"Attribute access to '{node.attr}' is not allowed "
                "(dunder access can escape the sandbox)."
            )

    return violations


def _check_import(root: str, full: str) -> list[str]:
    if root in BANNED_IMPORTS:
        return [f"Import of '{full}' is blocked by the sandbox policy."]
    if root not in ALLOWED_IMPORTS:
        return [
            f"Import of '{full}' is not on the allow-list "
            f"({', '.join(sorted(ALLOWED_IMPORTS))})."
        ]
    return []


def _is_dunder(name: str) -> bool:
    """True for `__x__`-style names, which are the usual sandbox-escape route."""
    return name.startswith("__") and name.endswith("__") and len(name) > 4
