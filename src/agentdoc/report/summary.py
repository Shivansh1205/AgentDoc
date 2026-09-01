"""The `ReportSummary` schema: a `ClassificationResult` reshaped for display.

Kept separate from `generator.py` (the logic that builds one) so both the
terminal renderer and the JSON exporter can depend on this shape without
pulling in generation logic, mirroring how `classifier.results` is kept
separate from `classifier.engine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentdoc.classifier.results import FlaggedFailure
from agentdoc.classifier.taxonomy import FailureCategory, FailureMode
from agentdoc.parsers.schema import Turn


@dataclass
class CategoryCount:
    """Failure count for one MAST category."""

    category: FailureCategory
    count: int


@dataclass
class FailureModeCount:
    """How many times one specific failure mode was flagged in this trace."""

    failure_mode: FailureMode
    count: int


@dataclass
class ReportSummary:
    """A `ClassificationResult` reshaped into a display- and export-ready report.

    Attributes:
        total_failures: Total number of flagged failures (== len(flagged_failures)).
        category_counts: Failure count per MAST category, in `FailureCategory`
            enum order, including categories with zero flagged failures (so
            consumers don't need to special-case missing keys).
        ranked_failure_modes: Distinct failure modes flagged in this trace,
            most frequent first; ties broken by the mode's paper ID for
            deterministic output.
        narrative: A short (1-3 sentence) plain-English summary of the
            overall diagnosis, generated from the counts above — see
            `agentdoc.report.generator.build_narrative`.
        flagged_failures: The full list of flagged failures, unmodified from
            the source `ClassificationResult`, for detailed drill-down.
        model: The LLM backend/model identifier used to produce the
            underlying classification, carried through for provenance.
        trace_turn_count: Number of turns in the source trace, for context
            (e.g. "4 failures across a 12-turn trace").
        source_framework: The trace's source framework (e.g. "langgraph").
        turns: The source trace's turns, carried through so exports can
            resolve each flagged failure's `turn_indices` against the actual
            turns they point at. Without these, a `turn_indices` value is a
            dangling reference: a consumer knows *that* turn 5 was faulted
            but has no way to learn which agent produced it or what it said.
            Empty by default so existing callers that build a summary without
            a trace keep working.
    """

    total_failures: int
    category_counts: list[CategoryCount]
    ranked_failure_modes: list[FailureModeCount]
    narrative: str
    flagged_failures: list[FlaggedFailure] = field(default_factory=list)
    model: str | None = None
    trace_turn_count: int = 0
    source_framework: str | None = None
    turns: list[Turn] = field(default_factory=list)
