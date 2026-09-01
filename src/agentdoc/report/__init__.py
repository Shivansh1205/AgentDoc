"""Report generation.

Turns a `ClassificationResult` (see `agentdoc.classifier.results`) into a
`ReportSummary` — counts by category, a ranked list of failure modes, a
rule-based plain-English narrative, and the full flagged-failure detail —
and renders it to the terminal (`agentdoc.report.terminal`), as JSON
(`agentdoc.report.json_export`), or as a self-contained HTML graph
visualization (`agentdoc.report.html`).
"""

from agentdoc.report.generator import build_narrative, generate_report
from agentdoc.report.html import Graph, GraphEdge, GraphNode, build_graph, render_html, write_report_html
from agentdoc.report.json_export import report_to_dict, report_to_json, write_report_json
from agentdoc.report.summary import CategoryCount, FailureModeCount, ReportSummary
from agentdoc.report.terminal import render_report

__all__ = [
    "generate_report",
    "build_narrative",
    "ReportSummary",
    "CategoryCount",
    "FailureModeCount",
    "render_report",
    "report_to_dict",
    "report_to_json",
    "write_report_json",
    "render_html",
    "write_report_html",
    "build_graph",
    "Graph",
    "GraphNode",
    "GraphEdge",
]
