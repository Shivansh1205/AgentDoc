"""Smoke tests for the agentdoc CLI."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentdoc import __version__
from agentdoc.classifier.results import ClassificationResult, FlaggedFailure
from agentdoc.classifier.taxonomy import FailureCategory, FailureMode
import agentdoc.cli as cli_module
from agentdoc.cli import app

runner = CliRunner()

EXAMPLE_TRACE = Path(__file__).parent.parent / "examples" / "langgraph_trace_example.json"


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_parse_command_renders_trace() -> None:
    result = runner.invoke(app, ["parse", str(EXAMPLE_TRACE)])
    assert result.exit_code == 0
    assert "Normalized trace" in result.stdout
    assert "researcher" in result.stdout
    assert "writer" in result.stdout
    assert "web_search" in result.stdout


def test_parse_command_unknown_framework() -> None:
    result = runner.invoke(app, ["parse", str(EXAMPLE_TRACE), "--framework", "autogen"])
    assert result.exit_code == 1
    assert "Unknown framework" in result.stdout


def test_parse_command_missing_file() -> None:
    result = runner.invoke(app, ["parse", "does_not_exist.json"])
    assert result.exit_code != 0


def test_parse_command_malformed_trace(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json", encoding="utf-8")

    result = runner.invoke(app, ["parse", str(bad_file)])
    assert result.exit_code == 1
    assert "Failed to parse trace" in result.stdout


class _FakeClassifier:
    """Stand-in for MastClassifier so `diagnose` tests never call a real LLM."""

    def __init__(self, result: ClassificationResult) -> None:
        self._result = result

    def classify(self, trace):
        return self._result


def test_diagnose_command_renders_flagged_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = ClassificationResult(
        flagged_failures=[
            FlaggedFailure(
                failure_mode=FailureMode.REASONING_ACTION_MISMATCH,
                category=FailureCategory.INTER_AGENT_MISALIGNMENT,
                turn_indices=[1, 2],
                justification="The agent contradicted its own tool result.",
                confidence=0.9,
            )
        ],
        model="fake-model-1",
    )
    monkeypatch.setattr(
        cli_module, "MastClassifier", lambda backend=None: _FakeClassifier(fake_result)
    )

    result = runner.invoke(app, ["diagnose", str(EXAMPLE_TRACE)])

    assert result.exit_code == 0
    assert "MAST Diagnosis Summary" in result.stdout
    assert "FM-2.6" in result.stdout
    assert "Reasoning-action mismatch" in result.stdout
    assert "contradicted its own tool result" in result.stdout


def test_diagnose_command_reports_no_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = ClassificationResult(flagged_failures=[], model="fake-model-1")
    monkeypatch.setattr(
        cli_module, "MastClassifier", lambda backend=None: _FakeClassifier(fake_result)
    )

    result = runner.invoke(app, ["diagnose", str(EXAMPLE_TRACE)])

    assert result.exit_code == 0
    assert "no issues were flagged" in result.stdout


def test_diagnose_command_missing_api_key_default_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default backend is "groq" (see agentdoc.classifier.engine.DEFAULT_BACKEND).
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = runner.invoke(app, ["diagnose", str(EXAMPLE_TRACE)])

    assert result.exit_code == 1
    assert "Cannot run classifier" in result.stdout
    assert "GROQ_API_KEY" in result.stdout


def test_diagnose_command_missing_api_key_anthropic_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(
        app, ["diagnose", str(EXAMPLE_TRACE), "--backend", "anthropic"]
    )

    assert result.exit_code == 1
    assert "Cannot run classifier" in result.stdout
    assert "ANTHROPIC_API_KEY" in result.stdout


def test_diagnose_command_unknown_backend() -> None:
    result = runner.invoke(app, ["diagnose", str(EXAMPLE_TRACE), "--backend", "openai"])
    assert result.exit_code == 1
    assert "Cannot run classifier" in result.stdout
    assert "Unknown LLM backend" in result.stdout


def test_diagnose_command_unknown_framework() -> None:
    result = runner.invoke(app, ["diagnose", str(EXAMPLE_TRACE), "--framework", "autogen"])
    assert result.exit_code == 1
    assert "Unknown framework" in result.stdout


def test_diagnose_command_json_writes_file_and_still_prints_terminal_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    out_path = tmp_path / "report.json"

    result = runner.invoke(
        app, ["diagnose", str(EXAMPLE_TRACE), "--json", str(out_path)]
    )

    assert result.exit_code == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["total_failures"] == 1
    assert data["flagged_failures"][0]["failure_mode"] == "FM-2.6"
    # Terminal report should still print by default alongside --json.
    assert "MAST Diagnosis Summary" in result.stdout
    assert "Wrote JSON report to" in result.stdout
    assert out_path.name in result.stdout


def test_diagnose_command_json_only_suppresses_terminal_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_result = ClassificationResult(flagged_failures=[], model="fake-model-1")
    monkeypatch.setattr(
        cli_module, "MastClassifier", lambda backend=None: _FakeClassifier(fake_result)
    )
    out_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["diagnose", str(EXAMPLE_TRACE), "--json", str(out_path), "--json-only"],
    )

    assert result.exit_code == 0
    assert out_path.exists()
    assert "MAST Diagnosis Summary" not in result.stdout
    assert "Wrote JSON report" not in result.stdout
    assert result.stdout == ""


def test_diagnose_command_json_only_without_json_path_errors() -> None:
    result = runner.invoke(app, ["diagnose", str(EXAMPLE_TRACE), "--json-only"])

    assert result.exit_code == 1
    assert "--json-only requires --json" in result.stdout


def test_diagnose_command_json_export_matches_terminal_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The JSON export must carry the same underlying data as what's rendered
    # to the terminal, not a lossy subset.
    fake_result = ClassificationResult(
        flagged_failures=[
            FlaggedFailure(
                failure_mode=FailureMode.STEP_REPETITION,
                category=FailureCategory.SYSTEM_DESIGN,
                turn_indices=[0, 1],
                justification="Repeated the lookup unnecessarily.",
                confidence=0.7,
            )
        ],
        model="fake-model-1",
    )
    monkeypatch.setattr(
        cli_module, "MastClassifier", lambda backend=None: _FakeClassifier(fake_result)
    )
    out_path = tmp_path / "report.json"

    result = runner.invoke(
        app, ["diagnose", str(EXAMPLE_TRACE), "--json", str(out_path)]
    )

    assert result.exit_code == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["flagged_failures"][0]["justification"] == "Repeated the lookup unnecessarily."
    assert "Repeated the lookup unnecessarily." in result.stdout
