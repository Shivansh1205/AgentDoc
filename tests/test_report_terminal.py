"""Tests for `agentdoc.report.terminal` (Rich terminal rendering).

Exercises rendering through a real `Console` writing to an in-memory buffer,
including a fixture with Unicode-heavy justification text (smart quotes, em
dash, non-breaking hyphen) — the same class of content that previously
crashed on a legacy Windows (cp1252) console. See `agentdoc/cli.py`'s
stdout/stderr `reconfigure(errors="replace")` fix; here we additionally
force the console itself to cp1252 with `errors="replace"` so the test would
fail if that class of bug ever regressed, independent of the real terminal
this suite happens to run in.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from agentdoc.classifier.results import ClassificationResult, FlaggedFailure
from agentdoc.classifier.taxonomy import FailureCategory, FailureMode
from agentdoc.parsers.schema import NormalizedTrace, Role, Turn
from agentdoc.report.generator import generate_report
from agentdoc.report.terminal import render_report


def _cp1252_console() -> tuple[Console, io.TextIOWrapper]:
    """A Rich Console writing through a cp1252-encoded buffer with
    errors="replace", to reproduce the legacy-Windows-console encoding
    constraints under test regardless of the host OS/terminal."""
    raw = io.BytesIO()
    text_stream = io.TextIOWrapper(raw, encoding="cp1252", errors="replace")
    console = Console(file=text_stream, legacy_windows=False, safe_box=False, width=100)
    return console, text_stream


def _read(text_stream: io.TextIOWrapper, raw_buffer: io.BytesIO | None = None) -> str:
    text_stream.flush()
    text_stream.seek(0)
    return text_stream.read()


@pytest.fixture
def sample_trace() -> NormalizedTrace:
    return NormalizedTrace(
        source_framework="langgraph",
        turns=[
            Turn(step=0, role=Role.HUMAN, content="What is the spec version?"),
            Turn(
                step=1,
                role=Role.AGENT,
                agent="researcher",
                content="The spec is version 1.0.",
            ),
        ],
    )


def _failure(justification: str, turns: list[int] | None = None) -> FlaggedFailure:
    return FlaggedFailure(
        failure_mode=FailureMode.REASONING_ACTION_MISMATCH,
        category=FailureCategory.INTER_AGENT_MISALIGNMENT,
        turn_indices=turns or [1],
        justification=justification,
        confidence=0.85,
    )


def test_render_report_no_failures_does_not_crash(sample_trace: NormalizedTrace) -> None:
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(sample_trace, result)

    console, stream = _cp1252_console()
    render_report(console, summary, sample_trace)
    output = _read(stream)

    assert "MAST Diagnosis Summary" in output
    assert "no issues were flagged" in output


def test_render_report_with_failures_does_not_crash(sample_trace: NormalizedTrace) -> None:
    result = ClassificationResult(
        flagged_failures=[_failure("The agent contradicted its own tool result.")],
        model="test-model",
    )
    summary = generate_report(sample_trace, result)

    console, stream = _cp1252_console()
    render_report(console, summary, sample_trace)
    output = _read(stream)

    assert "FM-2.6" in output
    assert "Reasoning-action mismatch" in output
    assert "contradicted its own tool result" in output


def test_render_report_includes_offending_turn_content(
    sample_trace: NormalizedTrace,
) -> None:
    result = ClassificationResult(
        flagged_failures=[_failure("Contradiction found.", turns=[1])],
        model="test-model",
    )
    summary = generate_report(sample_trace, result)

    console, stream = _cp1252_console()
    render_report(console, summary, sample_trace)
    output = _read(stream)

    # The offending turn's own content should be surfaced inline for
    # legibility without cross-referencing a separate `parse` output.
    assert "The spec is version 1.0." in output
    assert "researcher" in output


def test_render_report_without_trace_still_renders(sample_trace: NormalizedTrace) -> None:
    # `trace` is optional; omitting it should degrade gracefully (no offending
    # turn content shown) rather than crash.
    result = ClassificationResult(
        flagged_failures=[_failure("Some justification.")], model="test-model"
    )
    summary = generate_report(sample_trace, result)

    console, stream = _cp1252_console()
    render_report(console, summary, trace=None)
    output = _read(stream)

    assert "FM-2.6" in output
    assert "Some justification." in output


def test_render_report_unicode_heavy_justification_does_not_crash(
    sample_trace: NormalizedTrace,
) -> None:
    # The exact class of content that previously crashed: smart quotes, an
    # em dash, and a non-breaking hyphen (U+2011) from LLM-generated text.
    unicode_justification = (
        "The agent’s response contains a non‑breaking hyphen "
        "and “smart quotes” plus an em dash — here, "
        "as well as emoji \U0001F6A8 and CJK 日本語."
    )
    result = ClassificationResult(
        flagged_failures=[_failure(unicode_justification)], model="test-model"
    )
    summary = generate_report(sample_trace, result)

    console, stream = _cp1252_console()
    # Must not raise UnicodeEncodeError.
    render_report(console, summary, sample_trace)
    output = _read(stream)

    assert "FM-2.6" in output
    # Unencodable characters degrade to replacement characters rather than
    # crashing; the surrounding ASCII text must still be intact.
    assert "non" in output and "breaking hyphen" in output
    assert "smart quotes" in output


def test_render_report_multiple_categories_all_sections_render(
    sample_trace: NormalizedTrace,
) -> None:
    failures = [
        FlaggedFailure(
            failure_mode=FailureMode.STEP_REPETITION,
            category=FailureCategory.SYSTEM_DESIGN,
            turn_indices=[0],
            justification="Repeated the same lookup twice.",
            confidence=0.7,
        ),
        FlaggedFailure(
            failure_mode=FailureMode.REASONING_ACTION_MISMATCH,
            category=FailureCategory.INTER_AGENT_MISALIGNMENT,
            turn_indices=[1],
            justification="Contradicted its own tool result.",
            confidence=0.9,
        ),
        FlaggedFailure(
            failure_mode=FailureMode.NO_OR_INCOMPLETE_VERIFICATION,
            category=FailureCategory.TASK_VERIFICATION,
            turn_indices=[1],
            justification="Never checked the final answer.",
            confidence=0.6,
        ),
    ]
    result = ClassificationResult(flagged_failures=failures, model="test-model")
    summary = generate_report(sample_trace, result)

    console, stream = _cp1252_console()
    render_report(console, summary, sample_trace)
    output = _read(stream)

    assert "System Design Issues" in output
    assert "Inter-Agent Misalignment" in output
    assert "Task Verification" in output
    assert "Repeated the same lookup twice." in output
    assert "Contradicted its own tool result." in output
    assert "Never checked the final answer." in output
