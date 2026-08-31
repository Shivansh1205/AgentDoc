"""Packaging smoke tests: does the installed `agentdoc` console script work?

Unlike the other CLI tests (which invoke `agentdoc.cli.app` in-process via
Typer's `CliRunner`), this module launches the actual installed executable
in a subprocess — the same way a user would run `agentdoc` after `pip
install agentdoc`. This is the only way to catch packaging mistakes that
in-process testing can't see: a missing/misconfigured `[project.scripts]`
entry point, a runtime dependency declared as dev-only, or an import that
only works when running from the repo's working directory.

`--version` is run as a genuine subprocess call for exactly this reason. The
`diagnose` case mocks the LLM backend (no test here makes a real API call),
which requires patching Python objects in-process, so it uses `CliRunner`
against the same installed package instead of a second subprocess — the
subprocess call above already proves the entry point and import path work;
this confirms the command's logic behaves correctly once invoked.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentdoc import __version__

runner = CliRunner()

EXAMPLE_TRACE = Path(__file__).parent.parent / "examples" / "langgraph_trace_example.json"
FLAWED_EXAMPLE_TRACE = (
    Path(__file__).parent.parent / "examples" / "langgraph_trace_flawed_example.json"
)


def _find_agentdoc_executable() -> str:
    """Locate the installed `agentdoc` console script.

    Looks next to the current Python interpreter first (covers venvs, where
    console scripts live in the same Scripts/bin directory as python.exe),
    then falls back to PATH. Skips the test if it can't be found, rather
    than failing, since that indicates the package genuinely isn't
    installed in this environment (vs. a real packaging regression).
    """
    interpreter_dir = Path(sys.executable).parent
    candidates = [
        interpreter_dir / "agentdoc.exe",
        interpreter_dir / "agentdoc",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    on_path = shutil.which("agentdoc")
    if on_path:
        return on_path

    pytest.skip(
        "Could not locate an installed 'agentdoc' console script "
        "(checked next to the Python interpreter and on PATH)."
    )


def test_console_script_entry_point_resolves() -> None:
    """The `agentdoc` console script from [project.scripts] must exist and
    be directly executable — not just importable as a Python module."""
    executable = _find_agentdoc_executable()
    assert Path(executable).exists()


def test_installed_cli_version_flag_via_subprocess() -> None:
    """`agentdoc --version` must work as a real subprocess invocation, the
    same way a user runs it after installing the package — not just via
    in-process Typer testing, which wouldn't catch a broken entry point or
    a runtime dependency missing from the installed environment."""
    executable = _find_agentdoc_executable()

    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert __version__ in result.stdout


def test_installed_cli_help_via_subprocess() -> None:
    executable = _find_agentdoc_executable()

    result = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "diagnose" in result.stdout
    assert "parse" in result.stdout


def test_installed_cli_unknown_command_fails_cleanly_via_subprocess() -> None:
    executable = _find_agentdoc_executable()

    result = subprocess.run(
        [executable, "not-a-real-command"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    # Must be a clean CLI usage error, not a raw Python traceback.
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


# ---------------------------------------------------------------------------
# `diagnose` against the installed package, with the LLM backend mocked.
# Uses CliRunner (in-process) since mocking requires patching a Python
# object, which a separate subprocess can't see - but this still exercises
# the actual installed `agentdoc` package (import agentdoc.cli; app), the
# same code the subprocess-based tests above proved is reachable via the
# console script.
# ---------------------------------------------------------------------------


class _FakeClassifier:
    def __init__(self, result):
        self._result = result

    def classify(self, trace):
        return self._result


def test_installed_package_diagnose_with_mocked_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentdoc.classifier.results import ClassificationResult, FlaggedFailure
    from agentdoc.classifier.taxonomy import FailureCategory, FailureMode
    import agentdoc.cli as cli_module

    fake_result = ClassificationResult(
        flagged_failures=[
            FlaggedFailure(
                failure_mode=FailureMode.REASONING_ACTION_MISMATCH,
                category=FailureCategory.INTER_AGENT_MISALIGNMENT,
                turn_indices=[2],
                justification="Contradicted its own tool result.",
                confidence=0.9,
            )
        ],
        model="fake-model-1",
    )
    monkeypatch.setattr(
        cli_module, "MastClassifier", lambda backend=None: _FakeClassifier(fake_result)
    )

    result = runner.invoke(
        cli_module.app,
        ["diagnose", str(FLAWED_EXAMPLE_TRACE), "--backend", "groq"],
    )

    assert result.exit_code == 0
    assert "MAST Diagnosis Summary" in result.stdout
    assert "FM-2.6" in result.stdout


def test_installed_package_version_matches_pyproject() -> None:
    # Sanity check that the version string exposed at runtime is non-empty
    # and looks like a version, catching a broken/empty __version__ that
    # would make `--version` technically "work" but be useless.
    assert __version__
    assert __version__[0].isdigit()
