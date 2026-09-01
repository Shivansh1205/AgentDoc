"""JSON serialization for `ReportSummary`.

Produces the *full* structured data (all flagged failures with their turn
references and justifications, plus counts, the narrative, and the trace's
turns themselves) — not just the narrative sentence — so this is usable
programmatically (dashboards, CI gates, further analysis) without
re-deriving anything from the terminal report.

Turns are included so a failure's `turn_indices` resolves: `[5]` on its own
tells a consumer a turn was faulted but nothing about who produced it or
what happened, which is not enough to draw a timeline or an agent graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentdoc.parsers.schema import Turn
from agentdoc.report.summary import ReportSummary

#: Bump if the JSON export shape changes in a way that could break consumers
#: (renamed/removed field, changed type). Additive changes (new optional
#: field) don't require a bump.
#:
#: v2 added the top-level `turns` array. Strictly additive, but bumped anyway:
#: consumers keying off the version to decide whether they can render a
#: timeline/graph need to distinguish "no turns because the trace was empty"
#: from "no turns because this file predates the field".
JSON_SCHEMA_VERSION = 2


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
        "turns": [_turn_to_dict(turn) for turn in summary.turns],
    }


def _turn_to_dict(turn: Turn) -> dict[str, Any]:
    """Convert one `Turn` into a JSON-serializable dict.

    Mirrors the `Turn` dataclass field-for-field rather than trimming to what
    any one consumer needs today: the whole point of exporting turns is that
    `turn_indices` on a flagged failure resolves to something complete enough
    to render (who acted, what they said, what they called, where control
    went next).
    """
    return {
        "step": turn.step,
        "role": turn.role.value,
        "agent": turn.agent,
        "content": turn.content,
        "tool_calls": [
            {
                "name": call.name,
                "call_id": call.call_id,
                "args": call.args,
                "result": call.result,
                "error": call.error,
            }
            for call in turn.tool_calls
        ],
        "timestamp": turn.timestamp,
        "parent_step": turn.parent_step,
        "handoff_to": turn.handoff_to,
        "metadata": turn.metadata,
    }


def report_to_json(summary: ReportSummary, *, indent: int = 2) -> str:
    """Serialize a `ReportSummary` to a JSON string."""
    return json.dumps(report_to_dict(summary), indent=indent, ensure_ascii=False)


def write_report_json(summary: ReportSummary, path: Path) -> None:
    """Write a `ReportSummary` as JSON to `path` (UTF-8, creates/overwrites)."""
    Path(path).write_text(report_to_json(summary), encoding="utf-8")
