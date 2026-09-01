"""Tests for `agentdoc.report.json_export`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdoc.classifier.results import ClassificationResult, FlaggedFailure
from agentdoc.classifier.taxonomy import FailureCategory, FailureMode
from agentdoc.parsers.schema import NormalizedTrace, Role, ToolCall, Turn
from agentdoc.report.generator import generate_report
from agentdoc.report.json_export import (
    JSON_SCHEMA_VERSION,
    report_to_dict,
    report_to_json,
    write_report_json,
)


@pytest.fixture
def sample_trace() -> NormalizedTrace:
    return NormalizedTrace(
        source_framework="langgraph",
        turns=[
            Turn(step=0, role=Role.HUMAN, content="hello"),
            Turn(step=1, role=Role.AGENT, agent="researcher", content="hi"),
        ],
    )


@pytest.fixture
def sample_result() -> ClassificationResult:
    return ClassificationResult(
        flagged_failures=[
            FlaggedFailure(
                failure_mode=FailureMode.REASONING_ACTION_MISMATCH,
                category=FailureCategory.INTER_AGENT_MISALIGNMENT,
                turn_indices=[0, 1],
                justification="Contradicted its own tool result.",
                confidence=0.9,
            ),
            FlaggedFailure(
                failure_mode=FailureMode.NO_OR_INCOMPLETE_VERIFICATION,
                category=FailureCategory.TASK_VERIFICATION,
                turn_indices=[1],
                justification="Never checked the output.",
                confidence=0.5,
            ),
        ],
        model="test-model",
    )


def test_report_to_dict_has_expected_top_level_keys(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    summary = generate_report(sample_trace, sample_result)
    data = report_to_dict(summary)

    expected_keys = {
        "schema_version",
        "model",
        "source_framework",
        "trace_turn_count",
        "total_failures",
        "narrative",
        "category_counts",
        "ranked_failure_modes",
        "flagged_failures",
        "turns",
    }
    assert set(data.keys()) == expected_keys
    assert data["schema_version"] == JSON_SCHEMA_VERSION


def test_report_to_dict_matches_summary_values(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    summary = generate_report(sample_trace, sample_result)
    data = report_to_dict(summary)

    assert data["model"] == "test-model"
    assert data["source_framework"] == "langgraph"
    assert data["trace_turn_count"] == 2
    assert data["total_failures"] == 2


def test_report_to_dict_flagged_failures_is_full_structured_data(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    # The JSON export must carry full detail (turn refs + justifications),
    # not just the narrative — this is the "programmatic use" contract.
    summary = generate_report(sample_trace, sample_result)
    data = report_to_dict(summary)

    assert len(data["flagged_failures"]) == 2
    first = data["flagged_failures"][0]
    assert first["failure_mode"] == "FM-2.6"
    assert first["category"] == "inter_agent_misalignment"
    assert first["turn_indices"] == [0, 1]
    assert first["justification"] == "Contradicted its own tool result."
    assert first["confidence"] == pytest.approx(0.9)


def test_report_to_dict_category_counts_cover_all_three_categories(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    summary = generate_report(sample_trace, sample_result)
    data = report_to_dict(summary)

    categories = {cc["category"] for cc in data["category_counts"]}
    assert categories == {c.value for c in FailureCategory}


def test_report_to_dict_ranked_failure_modes_shape(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    summary = generate_report(sample_trace, sample_result)
    data = report_to_dict(summary)

    assert all({"failure_mode", "count"} == set(item.keys()) for item in data["ranked_failure_modes"])


def test_report_to_json_is_valid_json_and_round_trips(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    summary = generate_report(sample_trace, sample_result)
    json_str = report_to_json(summary)

    parsed = json.loads(json_str)
    assert parsed == report_to_dict(summary)


def test_report_to_json_preserves_unicode_readably(
    sample_trace: NormalizedTrace,
) -> None:
    # ensure_ascii=False: real Unicode characters should appear directly in
    # the JSON text (not \uXXXX escapes), since JSON files are UTF-8 and
    # unrelated to the terminal-encoding constraints of `report/terminal.py`.
    unicode_justification = "The agent’s tool call returned “v3.2” — but it said v1.0."
    result = ClassificationResult(
        flagged_failures=[
            FlaggedFailure(
                failure_mode=FailureMode.REASONING_ACTION_MISMATCH,
                category=FailureCategory.INTER_AGENT_MISALIGNMENT,
                turn_indices=[1],
                justification=unicode_justification,
                confidence=0.9,
            )
        ]
    )
    summary = generate_report(sample_trace, result)
    json_str = report_to_json(summary)

    assert "\\u2019" not in json_str  # not escaped
    assert unicode_justification in json_str
    # Still valid, parseable JSON.
    parsed = json.loads(json_str)
    assert parsed["flagged_failures"][0]["justification"] == unicode_justification


def test_write_report_json_writes_utf8_file(
    tmp_path: Path, sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    summary = generate_report(sample_trace, sample_result)
    out_path = tmp_path / "report.json"

    write_report_json(summary, out_path)

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed["total_failures"] == 2


def test_write_report_json_overwrites_existing_file(
    tmp_path: Path, sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    out_path = tmp_path / "report.json"
    out_path.write_text("stale content", encoding="utf-8")

    summary = generate_report(sample_trace, sample_result)
    write_report_json(summary, out_path)

    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content["total_failures"] == 2


def test_empty_result_json_export_has_empty_failures_list(
    sample_trace: NormalizedTrace,
) -> None:
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(sample_trace, result)
    data = report_to_dict(summary)

    assert data["flagged_failures"] == []
    assert data["ranked_failure_modes"] == []
    assert data["total_failures"] == 0


def test_turns_are_exported_with_full_detail(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    summary = generate_report(sample_trace, sample_result)
    data = report_to_dict(summary)

    assert len(data["turns"]) == 2
    human, agent = data["turns"]
    assert human == {
        "step": 0,
        "role": "human",
        "agent": None,
        "content": "hello",
        "tool_calls": [],
        "timestamp": None,
        "parent_step": None,
        "handoff_to": None,
        "metadata": {},
    }
    assert agent["role"] == "agent"
    assert agent["agent"] == "researcher"


def test_exported_turn_count_matches_trace_turn_count(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    """`turns` and `trace_turn_count` must not be able to disagree."""
    data = report_to_dict(generate_report(sample_trace, sample_result))

    assert len(data["turns"]) == data["trace_turn_count"]


def test_flagged_failure_turn_indices_resolve_against_exported_turns(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    """The point of exporting turns: `turn_indices` must not dangle.

    A consumer drawing a timeline or agent graph looks up each flagged
    failure's turn indices in `turns`; if those don't resolve, the export
    tells you *that* something failed but not who or what.
    """
    data = report_to_dict(generate_report(sample_trace, sample_result))
    steps = {turn["step"] for turn in data["turns"]}

    for failure in data["flagged_failures"]:
        assert failure["turn_indices"], "expected at least one referenced turn"
        for index in failure["turn_indices"]:
            assert index in steps, f"turn_indices {index} does not resolve"


def test_tool_calls_and_handoffs_survive_export() -> None:
    """Tool calls and handoffs are what the graph draws; they must round-trip."""
    trace = NormalizedTrace(
        source_framework="langgraph",
        turns=[
            Turn(
                step=0,
                role=Role.AGENT,
                agent="flight_assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        name="search_flights",
                        call_id="fc_1",
                        args={"to": "JFK"},
                        result="[]",
                    )
                ],
                handoff_to="hotel_assistant",
            )
        ],
    )
    result = ClassificationResult(flagged_failures=[], model="test-model")

    data = report_to_dict(generate_report(trace, result))
    turn = data["turns"][0]

    assert turn["handoff_to"] == "hotel_assistant"
    assert turn["tool_calls"] == [
        {
            "name": "search_flights",
            "call_id": "fc_1",
            "args": {"to": "JFK"},
            "result": "[]",
            "error": None,
        }
    ]


def test_json_string_round_trips_turns(
    sample_trace: NormalizedTrace, sample_result: ClassificationResult
) -> None:
    summary = generate_report(sample_trace, sample_result)
    parsed = json.loads(report_to_json(summary))

    assert [turn["step"] for turn in parsed["turns"]] == [0, 1]
