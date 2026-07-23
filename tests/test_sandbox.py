"""The sandbox is the part that runs code an LLM wrote. It gets the most tests.

Each blocked-code test asserts on *why* it was blocked, not just that it failed —
a test that passes because of a typo in the snippet would be worse than no test.
"""

from __future__ import annotations

import pytest

from src.tools.sandbox import run_code
from src.tools.sandbox_policy import validate_code


class TestAllowedCode:
    def test_runs_and_returns_result(self, tmp_path):
        result = run_code(
            "import pandas as pd\n"
            "df = pd.DataFrame({'a': [1, 2, 3]})\n"
            "result = {'total': int(df['a'].sum())}",
            tmp_path,
        )
        assert result.ok, result.traceback
        assert result.result == {"total": 6}

    def test_captures_stdout(self, tmp_path):
        result = run_code("print('hello from the sandbox')\nresult = 1", tmp_path)
        assert result.ok
        assert "hello from the sandbox" in result.stdout

    def test_sklearn_is_available(self, tmp_path):
        result = run_code(
            "from sklearn.linear_model import LinearRegression\n"
            "import numpy as np\n"
            "m = LinearRegression().fit(np.array([[1],[2],[3]]), np.array([2,4,6]))\n"
            "result = round(float(m.coef_[0]), 3)",
            tmp_path,
        )
        assert result.ok, result.traceback
        assert result.result == 2.0

    def test_writes_inside_run_dir_are_allowed(self, tmp_path):
        result = run_code(
            "import pandas as pd\n"
            "pd.DataFrame({'a': [1]}).to_csv('clean.csv', index=False)\n"
            "result = 'saved'",
            tmp_path,
        )
        assert result.ok, result.traceback
        assert (tmp_path / "clean.csv").exists()

    def test_new_files_are_reported_as_artifacts(self, tmp_path):
        result = run_code("open('out.txt', 'w').write('x')\nresult = 1", tmp_path)
        assert result.ok
        assert [p.name for p in result.artifacts] == ["out.txt"]


class TestBlockedCode:
    @pytest.mark.parametrize(
        ("snippet", "expected"),
        [
            ("import os; os.system('whoami')", "os"),
            ("import subprocess; subprocess.run(['ls'])", "subprocess"),
            ("import sys; sys.exit(1)", "sys"),
            ("import socket; socket.socket()", "socket"),
            ("import requests; requests.get('http://evil.test')", "requests"),
            ("import pickle; pickle.loads(b'')", "pickle"),
            ("import shutil; shutil.rmtree('/')", "shutil"),
            ("from pathlib import Path; Path('/').unlink()", "pathlib"),
        ],
    )
    def test_blocked_imports_never_execute(self, tmp_path, snippet, expected):
        result = run_code(snippet, tmp_path)
        assert not result.ok
        assert any(expected in v for v in result.violations), result.violations

    @pytest.mark.parametrize(
        "snippet",
        [
            "eval('1 + 1')",
            "exec('x = 1')",
            "compile('x=1', '<s>', 'exec')",
            "__import__('os').system('whoami')",
            "().__class__.__base__.__subclasses__()",
            "globals()['__builtins__']",
            # String-form dunder access via getattr. Before getattr was banned,
            # this walked object's subclasses to a module's globals and pulled the
            # already-imported `os` out of sys.modules — a confirmed full escape
            # that the AST dunder-attribute rule could not see.
            'getattr(getattr(0, "__class__"), "__base__").__subclasses__()',
            'getattr(0, "__class__")',
            'setattr(object, "x", 1)',
        ],
    )
    def test_escape_routes_are_rejected(self, tmp_path, snippet):
        result = run_code(snippet, tmp_path)
        assert not result.ok
        assert result.violations

    def test_getattr_sys_modules_escape_is_blocked(self, tmp_path):
        """The exact payload that reached os.listdir before the getattr ban."""
        payload = (
            'osmod=None\n'
            'for c in getattr(getattr(getattr(0,"__class__"),"__base__"),'
            '"__subclasses__")():\n'
            '    g=getattr(getattr(c,"__init__",None),"__globals__",None)\n'
            '    if isinstance(g,dict) and "sys" in g:\n'
            '        osmod=g["sys"].modules.get("os"); break\n'
            'result=osmod.listdir(".") if osmod else "blocked"\n'
        )
        result = run_code(payload, tmp_path)
        assert not result.ok, "sandbox escape executed"
        assert result.violations

    def test_the_malicious_snippet_from_the_spec(self, tmp_path):
        """import os; os.system(...) — the acceptance criterion, stated plainly."""
        result = run_code("import os; os.system('echo pwned')", tmp_path)
        assert not result.ok
        assert not result.stdout
        assert "os" in result.violations[0]

    def test_violations_are_reported_to_the_caller(self, tmp_path):
        result = run_code("import os", tmp_path)
        assert "rejected before execution" in result.error_feedback


class TestRuntimeGuards:
    def test_write_outside_run_dir_is_refused(self, tmp_path):
        target = tmp_path.parent / "escaped.txt"
        result = run_code(f"open({str(target)!r}, 'w').write('x')", tmp_path)
        assert not result.ok
        assert "writes are confined" in result.traceback
        assert not target.exists()

    def test_read_outside_run_dir_is_refused(self, tmp_path):
        """A legitimate-looking pandas read of a file outside the run directory.

        The open happens in pandas' frame, not the generated code's, so the
        import guard can't see it — only the path guard catches this.
        """
        secret = tmp_path / "secrets.csv"
        secret.write_text("api_key\nsk-live-do-not-read\n")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        result = run_code(
            f"import pandas as pd\ndf = pd.read_csv({str(secret)!r})\n"
            "result = df.to_dict()",
            run_dir,
        )
        assert not result.ok
        assert "outside the run directory" in result.traceback
        assert "sk-live-do-not-read" not in str(result.result)

    def test_read_inside_run_dir_is_allowed(self, tmp_path):
        """The guard must not break the normal case: reading the run's own data."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "raw.csv").write_text("a,b\n1,2\n3,4\n")

        result = run_code(
            "import pandas as pd\nresult = len(pd.read_csv('raw.csv'))", run_dir
        )
        assert result.ok, result.traceback
        assert result.result == 2

    def test_timeout_kills_a_spin_loop(self, tmp_path):
        result = run_code("while True:\n    pass", tmp_path, timeout_s=5)
        assert not result.ok
        assert result.timed_out
        assert "timeout" in result.error_feedback.lower()


class TestPolicy:
    def test_clean_code_has_no_violations(self):
        assert validate_code("import pandas as pd\nresult = 1") == []

    def test_syntax_errors_are_reported_not_raised(self):
        violations = validate_code("def broken(:\n  pass")
        assert violations and "Syntax error" in violations[0]

    def test_unknown_imports_are_rejected_by_default(self):
        violations = validate_code("import scipy")
        assert violations and "allow-list" in violations[0]
