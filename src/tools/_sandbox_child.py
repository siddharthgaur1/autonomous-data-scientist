"""Runner executed inside the sandbox subprocess. Not imported by the parent.

Launched as `python -I -B _sandbox_child.py <job.json>`. It deliberately imports
nothing from this project: the whole policy arrives in the job file, so the child
cannot be steered by anything on the project's import path.

Order matters here. Limits are applied, then the guards are installed, and only
then is the generated code compiled and run.
"""

from __future__ import annotations

import builtins
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

_real_import = builtins.__import__
_real_open = builtins.open


def _apply_limits(memory_mb: int, timeout_s: int) -> None:
    """Cap address space and CPU. POSIX only; see the note in sandbox.py."""
    try:
        import resource  # noqa: PLC0415 - unavailable on Windows, hence the guard
    except ImportError:
        return
    nbytes = memory_mb * 1024 * 1024
    for limit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
        try:
            resource.setrlimit(limit, (nbytes, nbytes))
        except (ValueError, OSError):
            pass
    try:
        # Backstop for the parent's wall-clock kill: a spin loop burns CPU and
        # would otherwise sit at 100% until the parent notices.
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s + 1))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, OSError):
        pass


def _install_import_guard(code_file: str, allowed: set[str], banned: set[str]) -> None:
    """Enforce the import allow-list for imports written in the generated code.

    Only imports whose calling frame is the generated code are checked. Nested
    imports made by pandas or sklearn are let through — otherwise the allow-list
    would have to enumerate every transitive dependency of scikit-learn.
    """

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        frame = sys._getframe(1)
        if frame.f_code.co_filename == code_file:
            root = name.split(".", 1)[0]
            if root in banned or root not in allowed:
                raise ImportError(
                    f"Import of '{name}' is blocked by the sandbox policy."
                )
        return _real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded


def _install_open_guard(run_dir: str, read_roots: list[str]) -> None:
    """Confine writes to the run directory and reads to known-safe roots.

    Unlike the import guard this applies to every frame, including library code.
    A library opening a path is not evidence the path is safe: `pd.read_csv` on
    a credentials file would otherwise sail through, since the open happens in
    pandas' frame rather than the generated code's.
    """
    import os.path as _p  # noqa: PLC0415 - the child never exposes `os` to user code

    run_root = _p.realpath(run_dir)
    safe_read_roots = [_p.realpath(r) for r in read_roots] + [run_root]

    def _under(path: str, root: str) -> bool:
        try:
            return _p.commonpath([_p.realpath(path), root]) == root
        except ValueError:  # different drives on Windows
            return False

    def guarded(file, mode="r", *args, **kwargs):  # noqa: A002
        target = file if isinstance(file, (str, bytes)) else getattr(file, "name", "")
        if isinstance(target, bytes):
            target = target.decode("utf-8", "replace")
        if not isinstance(target, str) or not target:
            raise PermissionError("Sandbox: open() needs a path.")

        writing = any(m in mode for m in ("w", "a", "x", "+"))
        if writing:
            if not _under(target, run_root):
                raise PermissionError(
                    f"Sandbox: writes are confined to the run directory, "
                    f"refused '{target}'."
                )
        elif not any(_under(target, root) for root in safe_read_roots):
            raise PermissionError(
                f"Sandbox: reads outside the run directory are refused: '{target}'."
            )
        return _real_open(file, mode, *args, **kwargs)

    builtins.open = guarded


def _jsonable(value: object) -> object:
    """Best-effort JSON coercion so a returned DataFrame doesn't kill the run."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return json.loads(json.dumps(to_dict(), default=str))
            except Exception:  # noqa: BLE001 - fall through to repr
                pass
        return repr(value)[:5000]


def main() -> int:
    job = json.loads(_real_open(sys.argv[1], encoding="utf-8").read())

    code_file: str = job["code_path"]
    source = _real_open(code_file, encoding="utf-8").read()

    _apply_limits(job["memory_mb"], job["timeout_s"])
    _install_import_guard(code_file, set(job["allowed"]), set(job["banned"]))
    _install_open_guard(job["run_dir"], job["read_roots"])

    out, err = io.StringIO(), io.StringIO()
    payload: dict[str, object] = {"ok": False, "result": None, "traceback": ""}
    namespace: dict[str, object] = {"__name__": "__sandbox__", "__file__": code_file}

    try:
        compiled = compile(source, code_file, "exec")
        with redirect_stdout(out), redirect_stderr(err):
            exec(compiled, namespace)  # noqa: S102 - the point of the module
        payload["ok"] = True
        payload["result"] = _jsonable(namespace.get("result"))
    except BaseException:  # noqa: BLE001 - MemoryError and SystemExit must be reported
        payload["traceback"] = traceback.format_exc(limit=20)

    payload["stdout"] = out.getvalue()[-20000:]
    payload["stderr"] = err.getvalue()[-20000:]

    with _real_open(job["result_path"], "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
