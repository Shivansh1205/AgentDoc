"""JSON serialization for `ReportSummary`.

Produces the *full* structured data (all flagged failures with their turn
references and justifications, plus counts and the narrative) — not just the
narrative sentence — so this is usable programmatically (dashboards, CI
gates, further analysis) without re-deriving anything from the terminal
report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentdoc.report.summary import ReportSummary

#: Bump if the JSON export shape changes in a way that could break consumers
#: (renamed/removed field, changed type). Additive changes (new optional
#: field) don't require a bump.
JSON_SCHEMA_VERSION = 1


def report_to_dict(summary: ReportSummary) -> dict[str, Any]:
    """Convert a `ReportSummary` into a plain JSON-serializable dict."""
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "model": summary.model,
        "source_framework": summary.source_framework,
        "trace_turn_count": summary.trace_turn_count,
        "total_failures": summary.total_failures,
        "narrative": summary.narrative,
        "category_counts": [
            {"category": cc.category.value, "count": cc.count}
            for cc in summary.category_counts
        ],
        "ranked_failure_modes": [
            {"failure_mode": fmc.failure_mode.value, "count": fmc.count}
            for fmc in summary.ranked_failure_modes
        ],
        "flagged_failures": [
            {
                "failure_mode": failure.failure_mode.value,
                "category": failure.category.value,
                "turn_indices": list(failure.turn_indices),
                "justification": failure.justification,
                "confidence": failure.confidence,
            }
            for failure in summary.flagged_failures
        ],
    }


def report_to_json(summary: ReportSummary, *, indent: int = 2) -> str:
    """Serialize a `ReportSummary` to a JSON string."""
    return json.dumps(report_to_dict(summary), indent=indent, ensure_ascii=False)


def write_report_json(summary: ReportSummary, path: Path) -> None:
    """Write a `ReportSummary` as JSON to `path` (UTF-8, creates/overwrites)."""
    Path(path).write_text(report_to_json(summary), encoding="utf-8")
