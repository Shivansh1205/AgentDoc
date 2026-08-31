"""Turns a `ClassificationResult` into a `ReportSummary`.

The narrative summary is deliberately template/rule-based off the failure
counts and categories — no additional LLM call. This keeps report generation
fast, free, and deterministic given a `ClassificationResult`.
"""

from __future__ import annotations

from collections import Counter

from agentdoc.classifier.results import ClassificationResult
from agentdoc.classifier.taxonomy import (
    CATEGORY_NAMES,
    FailureCategory,
    get_failure_mode,
)
from agentdoc.parsers.schema import NormalizedTrace
from agentdoc.report.summary import CategoryCount, FailureModeCount, ReportSummary


def generate_report(
    trace: NormalizedTrace, result: ClassificationResult
) -> ReportSummary:
    """Build a `ReportSummary` from a trace and its classification result."""
    category_counts = [
        CategoryCount(category=category, count=len(result.by_category(category)))
        for category in FailureCategory
    ]
    ranked_failure_modes = _rank_failure_modes(result)
    narrative = build_narrative(
        total_failures=len(result),
        category_counts=category_counts,
        ranked_failure_modes=ranked_failure_modes,
    )

    return ReportSummary(
        total_failures=len(result),
        category_counts=category_counts,
        ranked_failure_modes=ranked_failure_modes,
        narrative=narrative,
        flagged_failures=list(result.flagged_failures),
        model=result.model,
        trace_turn_count=len(trace),
        source_framework=trace.source_framework,
    )


def _rank_failure_modes(result: ClassificationResult) -> list[FailureModeCount]:
    """Count occurrences of each distinct failure mode, most frequent first.

    Ties are broken by the mode's paper ID (e.g. "FM-2.4" before "FM-3.1") so
    output is deterministic regardless of dict/Counter iteration order.
    """
    counts = Counter(failure.failure_mode for failure in result.flagged_failures)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].value))
    return [FailureModeCount(failure_mode=mode, count=count) for mode, count in ranked]


def build_narrative(
    *,
    total_failures: int,
    category_counts: list[CategoryCount],
    ranked_failure_modes: list[FailureModeCount],
) -> str:
    """Generate a short (1-3 sentence) plain-English diagnosis summary.

    Rule-based, not another LLM call: sentence 1 states the headline count
    (or a clean bill of health); sentence 2 names the dominant category (only
    when failures actually cluster there, i.e. it holds a clear plurality);
    sentence 3 calls out the single most frequent failure mode by name, with
    an occurrence count when it repeated.
    """
    if total_failures == 0:
        return "This run shows no MAST failure modes: no issues were flagged."

    failure_word = "failure" if total_failures == 1 else "failures"
    sentences = [f"This run shows {total_failures} {failure_word}."]

    nonzero_categories = [c for c in category_counts if c.count > 0]
    if len(nonzero_categories) > 1:
        dominant = max(nonzero_categories, key=lambda c: c.count)
        # Only call out a "primary driver" when it's a clear plurality, not a
        # near-even split across categories where no one category dominates.
        if dominant.count > total_failures / 2 or dominant.count >= 2 * (
            total_failures - dominant.count
        ):
            category_name = CATEGORY_NAMES[dominant.category]
            sentences.append(
                f"It is primarily driven by {category_name.lower()} "
                f"({dominant.count} of {total_failures})."
            )
    elif len(nonzero_categories) == 1:
        category_name = CATEGORY_NAMES[nonzero_categories[0].category]
        sentences.append(f"All flagged failures fall under {category_name.lower()}.")

    if ranked_failure_modes:
        top = ranked_failure_modes[0]
        mode_info = get_failure_mode(top.failure_mode.value)
        if top.count > 1:
            sentences.append(
                f"The most common issue is {mode_info.name.lower()} "
                f"({top.failure_mode.value}), occurring {top.count} times."
            )
        elif len(ranked_failure_modes) == 1:
            sentences.append(
                f"The issue identified is {mode_info.name.lower()} "
                f"({top.failure_mode.value})."
            )

    return " ".join(sentences)
