"""Tests for `agentdoc.report.generator` (ReportSummary + narrative logic)."""

from __future__ import annotations

import pytest

from agentdoc.classifier.results import ClassificationResult, FlaggedFailure
from agentdoc.classifier.taxonomy import FailureCategory, FailureMode
from agentdoc.parsers.schema import NormalizedTrace, Role, ToolCall, Turn
from agentdoc.report.generator import build_narrative, generate_report
from agentdoc.report.summary import CategoryCount, FailureModeCount


@pytest.fixture
def flawed_trace() -> NormalizedTrace:
    """Same shape as the classifier's flawed-trace fixture: an agent ignores
    a tool result and contradicts itself, then a second agent proceeds
    without seeking clarification."""
    return NormalizedTrace(
        source_framework="langgraph",
        turns=[
            Turn(step=0, role=Role.HUMAN, content="What is the current spec version?"),
            Turn(
                step=1,
                role=Role.AGENT,
                agent="researcher",
                content="Let me check.",
                tool_calls=[
                    ToolCall(
                        name="lookup_spec",
                        call_id="call_1",
                        result="Version 3.2, published last week.",
                    )
                ],
            ),
            Turn(
                step=2,
                role=Role.AGENT,
                agent="researcher",
                content="The spec is version 1.0, stable for years.",
            ),
            Turn(step=3, role=Role.AGENT, agent="writer", content="Drafting on v1.0."),
        ],
    )


def _failure(
    mode: FailureMode, category: FailureCategory, turns: list[int], conf: float = 0.8
) -> FlaggedFailure:
    return FlaggedFailure(
        failure_mode=mode,
        category=category,
        turn_indices=turns,
        justification=f"justification for {mode.value}",
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# generate_report: structure
# ---------------------------------------------------------------------------


def test_generate_report_with_no_failures(flawed_trace: NormalizedTrace) -> None:
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(flawed_trace, result)

    assert summary.total_failures == 0
    assert summary.ranked_failure_modes == []
    assert summary.flagged_failures == []
    assert summary.model == "test-model"
    assert summary.trace_turn_count == 4
    assert summary.source_framework == "langgraph"
    assert all(cc.count == 0 for cc in summary.category_counts)
    assert "no MAST failure modes" in summary.narrative or "no issues" in summary.narrative


def test_generate_report_includes_all_categories_even_with_zero_count(
    flawed_trace: NormalizedTrace,
) -> None:
    result = ClassificationResult(
        flagged_failures=[
            _failure(
                FailureMode.REASONING_ACTION_MISMATCH,
                FailureCategory.INTER_AGENT_MISALIGNMENT,
                [1, 2],
            )
        ]
    )
    summary = generate_report(flawed_trace, result)

    categories_present = {cc.category for cc in summary.category_counts}
    assert categories_present == set(FailureCategory)


def test_generate_report_category_counts_are_correct(
    flawed_trace: NormalizedTrace,
) -> None:
    result = ClassificationResult(
        flagged_failures=[
            _failure(
                FailureMode.REASONING_ACTION_MISMATCH,
                FailureCategory.INTER_AGENT_MISALIGNMENT,
                [2],
            ),
            _failure(
                FailureMode.FAIL_TO_ASK_FOR_CLARIFICATION,
                FailureCategory.INTER_AGENT_MISALIGNMENT,
                [3],
            ),
            _failure(
                FailureMode.NO_OR_INCOMPLETE_VERIFICATION,
                FailureCategory.TASK_VERIFICATION,
                [3],
            ),
        ]
    )
    summary = generate_report(flawed_trace, result)

    counts = {cc.category: cc.count for cc in summary.category_counts}
    assert counts[FailureCategory.INTER_AGENT_MISALIGNMENT] == 2
    assert counts[FailureCategory.TASK_VERIFICATION] == 1
    assert counts[FailureCategory.SYSTEM_DESIGN] == 0
    assert summary.total_failures == 3


def test_generate_report_preserves_flagged_failures_for_drilldown(
    flawed_trace: NormalizedTrace,
) -> None:
    failure = _failure(
        FailureMode.REASONING_ACTION_MISMATCH,
        FailureCategory.INTER_AGENT_MISALIGNMENT,
        [1, 2],
    )
    result = ClassificationResult(flagged_failures=[failure])
    summary = generate_report(flawed_trace, result)

    assert summary.flagged_failures == [failure]


# ---------------------------------------------------------------------------
# ranked_failure_modes
# ---------------------------------------------------------------------------


def test_ranked_failure_modes_orders_by_frequency_desc(
    flawed_trace: NormalizedTrace,
) -> None:
    result = ClassificationResult(
        flagged_failures=[
            _failure(FailureMode.STEP_REPETITION, FailureCategory.SYSTEM_DESIGN, [0]),
            _failure(FailureMode.STEP_REPETITION, FailureCategory.SYSTEM_DESIGN, [1]),
            _failure(FailureMode.STEP_REPETITION, FailureCategory.SYSTEM_DESIGN, [2]),
            _failure(
                FailureMode.PREMATURE_TERMINATION,
                FailureCategory.TASK_VERIFICATION,
                [3],
            ),
        ]
    )
    summary = generate_report(flawed_trace, result)

    assert summary.ranked_failure_modes[0].failure_mode == FailureMode.STEP_REPETITION
    assert summary.ranked_failure_modes[0].count == 3
    assert summary.ranked_failure_modes[1].failure_mode == FailureMode.PREMATURE_TERMINATION
    assert summary.ranked_failure_modes[1].count == 1


def test_ranked_failure_modes_breaks_ties_by_fm_id(flawed_trace: NormalizedTrace) -> None:
    # FM-2.6 and FM-3.1 both occur once; deterministic ordering by ID.
    result = ClassificationResult(
        flagged_failures=[
            _failure(
                FailureMode.PREMATURE_TERMINATION,
                FailureCategory.TASK_VERIFICATION,
                [0],
            ),
            _failure(
                FailureMode.REASONING_ACTION_MISMATCH,
                FailureCategory.INTER_AGENT_MISALIGNMENT,
                [1],
            ),
        ]
    )
    summary = generate_report(flawed_trace, result)

    ids_in_order = [fmc.failure_mode.value for fmc in summary.ranked_failure_modes]
    assert ids_in_order == ["FM-2.6", "FM-3.1"]


# ---------------------------------------------------------------------------
# build_narrative
# ---------------------------------------------------------------------------


def test_narrative_zero_failures() -> None:
    narrative = build_narrative(
        total_failures=0, category_counts=[], ranked_failure_modes=[]
    )
    assert "no" in narrative.lower()
    assert "MAST" in narrative or "issues" in narrative.lower()


def test_narrative_singular_vs_plural_failure_wording() -> None:
    counts = [CategoryCount(category=FailureCategory.SYSTEM_DESIGN, count=1)]
    ranked = [FailureModeCount(failure_mode=FailureMode.STEP_REPETITION, count=1)]

    singular = build_narrative(
        total_failures=1, category_counts=counts, ranked_failure_modes=ranked
    )
    assert "1 failure." in singular
    assert "1 failures" not in singular

    plural_counts = [CategoryCount(category=FailureCategory.SYSTEM_DESIGN, count=3)]
    plural_ranked = [FailureModeCount(failure_mode=FailureMode.STEP_REPETITION, count=3)]
    plural = build_narrative(
        total_failures=3, category_counts=plural_counts, ranked_failure_modes=plural_ranked
    )
    assert "3 failures." in plural


def test_narrative_names_dominant_category_when_clustered() -> None:
    counts = [
        CategoryCount(category=FailureCategory.SYSTEM_DESIGN, count=0),
        CategoryCount(category=FailureCategory.INTER_AGENT_MISALIGNMENT, count=4),
        CategoryCount(category=FailureCategory.TASK_VERIFICATION, count=1),
    ]
    ranked = [
        FailureModeCount(failure_mode=FailureMode.REASONING_ACTION_MISMATCH, count=2),
        FailureModeCount(failure_mode=FailureMode.FAIL_TO_ASK_FOR_CLARIFICATION, count=2),
        FailureModeCount(failure_mode=FailureMode.PREMATURE_TERMINATION, count=1),
    ]
    narrative = build_narrative(total_failures=5, category_counts=counts, ranked_failure_modes=ranked)

    assert "inter-agent misalignment" in narrative.lower()
    assert "4 of 5" in narrative


def test_narrative_names_top_failure_mode_with_count() -> None:
    counts = [CategoryCount(category=FailureCategory.SYSTEM_DESIGN, count=3)]
    ranked = [FailureModeCount(failure_mode=FailureMode.STEP_REPETITION, count=3)]
    narrative = build_narrative(total_failures=3, category_counts=counts, ranked_failure_modes=ranked)

    assert "step repetition" in narrative.lower()
    assert "FM-1.3" in narrative
    assert "3 times" in narrative


def test_narrative_single_failure_mode_singular_phrasing() -> None:
    counts = [CategoryCount(category=FailureCategory.TASK_VERIFICATION, count=1)]
    ranked = [FailureModeCount(failure_mode=FailureMode.PREMATURE_TERMINATION, count=1)]
    narrative = build_narrative(total_failures=1, category_counts=counts, ranked_failure_modes=ranked)

    assert "premature termination" in narrative.lower()
    # Singular case should not say "occurring 1 times".
    assert "1 times" not in narrative


def test_narrative_even_split_across_categories_does_not_claim_dominance() -> None:
    # 2 categories, evenly split (2 and 2) - no category holds a clear
    # plurality, so the narrative should not claim one "primarily drives" it.
    counts = [
        CategoryCount(category=FailureCategory.SYSTEM_DESIGN, count=2),
        CategoryCount(category=FailureCategory.INTER_AGENT_MISALIGNMENT, count=2),
    ]
    ranked = [
        FailureModeCount(failure_mode=FailureMode.STEP_REPETITION, count=2),
        FailureModeCount(failure_mode=FailureMode.TASK_DERAILMENT, count=2),
    ]
    narrative = build_narrative(total_failures=4, category_counts=counts, ranked_failure_modes=ranked)

    assert "primarily driven by" not in narrative.lower()


def test_narrative_is_one_to_three_sentences() -> None:
    counts = [CategoryCount(category=FailureCategory.SYSTEM_DESIGN, count=3)]
    ranked = [FailureModeCount(failure_mode=FailureMode.STEP_REPETITION, count=3)]
    narrative = build_narrative(total_failures=3, category_counts=counts, ranked_failure_modes=ranked)

    sentence_count = narrative.count(". ") + 1
    assert 1 <= sentence_count <= 3
