"""Self-contained HTML graph visualization of a `ReportSummary`.

Produces a single HTML file with inline CSS/JS and inline SVG - no external
server, build step, or network access needed to view it; just open it in a
browser. This is a third report renderer alongside `terminal.py` (Rich) and
`json_export.py` (JSON), following the same pattern: it consumes the same
`ReportSummary` (+ the source `NormalizedTrace` for turn/agent detail) and
invents no new fields — every value shown here already exists on
`ReportSummary`, `FlaggedFailure`, or `Turn`.

Layout approach
----------------
The data layer (`build_graph`, `GraphNode`, `GraphEdge`) derives one node per
distinct agent and one edge per turn-to-turn handoff, independent of any
rendering concern — see their docstrings. This module only decides how to
*draw* that structure. Python emits nodes/edges with no fixed position;
initial coordinates are placed on a circle (a reasonable starting guess with
no data dependency) and a small vanilla-JS force simulation — mutual
node repulsion, spring attraction along edges, gentle centering — settles
them into their final layout client-side as the page loads. This is a
handful of well-known physics equations, not a graph library: no dependency
is worth pulling in for what's usually under 20 nodes.

Node radius scales with `len(node.turn_steps)` (how many turns that agent
produced) so activity reads as size, not just position.

Tool calls are exposed as a `tools` field on the node's tooltip data rather
than a separate node type — a tool call happens *during* an agent's turn,
not as a distinct conversational actor, so it belongs to the node it
occurred on (see `agentdoc.parsers.schema.Turn.tool_calls`).

Interaction is hover-first: moving over a node or edge immediately shows its
detail (agent, activity, and any flagged failure with mode/category/
justification/confidence) in a floating tooltip that follows the pointer.
Clicking pins the tooltip open (for copy/paste or a steadier read) until
something else is hovered or the pin is dismissed.

Dependencies: none. Plain inline SVG + vanilla JS for both the physics and
the hover/pin interaction.
"""

from __future__ import annotations

import html as _html
import json
import math
from dataclasses import dataclass
from pathlib import Path

from agentdoc.classifier.results import FlaggedFailure
from agentdoc.classifier.taxonomy import CATEGORY_NAMES, FailureCategory, get_failure_mode
from agentdoc.parsers.schema import NormalizedTrace, Turn
from agentdoc.report.summary import ReportSummary

# --- Layout constants --------------------------------------------------------
#
# Initial positions only: nodes start on a circle (a deterministic, data-
# independent starting guess) and the client-side force simulation takes it
# from there. Canvas size and the starting circle's radius both scale with
# node count - a fixed canvas would leave a 2-node graph looking lost in a
# mostly-empty box (the physics settles at a fixed rest distance regardless
# of canvas size) and would crowd a 15-node graph into too little room.

_CANVAS_W_MIN, _CANVAS_W_MAX = 460, 900
_CANVAS_H_MIN, _CANVAS_H_MAX = 280, 560
_NODE_RADIUS_MIN = 22
_NODE_RADIUS_MAX = 40


def _canvas_size(node_count: int) -> tuple[float, float]:
    """Canvas dimensions sized to comfortably fit `node_count` nodes at
    their eventual settled spacing, clamped to a sane min/max."""
    if node_count <= 1:
        return _CANVAS_W_MIN, _CANVAS_H_MIN
    # ~190px of settled spread per additional node is a reasonable estimate
    # given the spring rest length/repulsion balance in the JS simulation.
    w = 320 + node_count * 95
    h = 220 + node_count * 55
    return (
        max(_CANVAS_W_MIN, min(_CANVAS_W_MAX, w)),
        max(_CANVAS_H_MIN, min(_CANVAS_H_MAX, h)),
    )


@dataclass(frozen=True)
class GraphNode:
    """One agent node in the visualization.

    Attributes:
        agent: The agent name (`Turn.agent`), used as the node's stable id.
        index: Position among distinct agents in order of first appearance
            (0-indexed) - determines x position.
        turn_steps: All `Turn.step` values this agent produced, in order.
        tool_names: Distinct tool names this agent invoked across the trace
            (from `Turn.tool_calls`), for the node's tool badge.
        failing: Whether any flagged failure implicates one of this node's
            turns.
        failures: The flagged failures implicating this node, for the
            click-to-inspect detail panel.
    """

    agent: str
    index: int
    turn_steps: list[int]
    tool_names: list[str]
    failing: bool
    failures: list[FlaggedFailure]


@dataclass(frozen=True)
class GraphEdge:
    """One directed edge representing a handoff between two agents.

    Attributes:
        source: Source node agent name.
        target: Target node agent name.
        from_step: The earlier turn's step.
        to_step: The later turn's step.
        is_loopback: True when `target` already appeared before `source`'s
            turn (i.e. this edge revisits an earlier agent), which is drawn
            as a curved arc above the row instead of a straight line.
        explicit: True when this edge comes from a structural handoff
            signal (`Turn.handoff_to`, e.g. a langgraph-swarm
            `transfer_to_<agent>` tool call) rather than being inferred from
            `Turn.agent` changing between consecutive turns. An explicit
            handoff is a framework telling us "control passed here"; an
            inferred one is us guessing from turn order. Rendered distinctly
            (see `report/html.py`'s `_JS`) so a real handoff reads as more
            than an ordinary tool-call badge.
        failing: Whether either endpoint turn is implicated by a flagged
            failure.
        failures: The flagged failures implicating this edge, for the
            click-to-inspect detail panel.
    """

    source: str
    target: str
    from_step: int
    to_step: int
    is_loopback: bool
    explicit: bool
    failing: bool
    failures: list[FlaggedFailure]


@dataclass(frozen=True)
class Graph:
    """The full graph derived from a trace + its flagged failures."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


def build_graph(trace: NormalizedTrace, flagged_failures: list[FlaggedFailure]) -> Graph:
    """Derive a deterministic agent graph from a trace and its failures.

    Turns with no `agent` (e.g. a raw system/human message with no owning
    agent) are skipped for graph purposes — they carry no node identity to
    attach to — but are still fully represented in the terminal/JSON
    reports; this only affects the graph visualization.

    Edges come from whichever handoff signal the trace actually provides:
    if any turn carries an explicit `Turn.handoff_to` (e.g. a
    langgraph-swarm `transfer_to_<agent>` tool call), every edge in the
    graph is built from that structural signal (`_build_explicit_edges`).
    Otherwise, edges are inferred from `Turn.agent` changing between
    consecutive turns (`_build_inferred_edges`), exactly as before
    `handoff_to` existed. The two strategies are never mixed within one
    graph, since that could draw the same transition twice or disagree
    about where an edge points.
    """
    agent_turns = [turn for turn in trace if turn.agent]

    # Map each Turn.step -> the set of failures citing it, for O(1) lookup
    # while building nodes/edges instead of re-scanning flagged_failures.
    failures_by_step: dict[int, list[FlaggedFailure]] = {}
    for failure in flagged_failures:
        for step in failure.turn_indices:
            failures_by_step.setdefault(step, []).append(failure)

    nodes_by_agent: dict[str, GraphNode] = {}
    node_order: list[str] = []

    for turn in agent_turns:
        agent = turn.agent
        assert agent is not None  # filtered above
        turn_failures = failures_by_step.get(turn.step, [])

        if agent not in nodes_by_agent:
            node_order.append(agent)
            nodes_by_agent[agent] = GraphNode(
                agent=agent,
                index=len(node_order) - 1,
                turn_steps=[turn.step],
                tool_names=[call.name for call in turn.tool_calls],
                failing=bool(turn_failures),
                failures=list(turn_failures),
            )
        else:
            existing = nodes_by_agent[agent]
            merged_tools = existing.tool_names + [
                call.name for call in turn.tool_calls if call.name not in existing.tool_names
            ]
            nodes_by_agent[agent] = GraphNode(
                agent=existing.agent,
                index=existing.index,
                turn_steps=[*existing.turn_steps, turn.step],
                tool_names=merged_tools,
                failing=existing.failing or bool(turn_failures),
                failures=_merge_failures(existing.failures, turn_failures),
            )

    has_explicit_handoffs = any(turn.handoff_to for turn in agent_turns)
    if has_explicit_handoffs:
        edges = _build_explicit_edges(agent_turns, failures_by_step)
    else:
        edges = _build_inferred_edges(agent_turns, failures_by_step)

    return Graph(nodes=[nodes_by_agent[a] for a in node_order], edges=edges)


def _build_explicit_edges(
    agent_turns: list[Turn], failures_by_step: dict[int, list[FlaggedFailure]]
) -> list[GraphEdge]:
    """Build edges from structural `Turn.handoff_to` signals.

    Used for the whole graph once *any* turn in the trace carries an
    explicit handoff - mixing explicit and inferred edge-generation
    strategies in one graph would risk drawing the same transition twice
    (once from each strategy) or disagreeing about where an edge points.

    Each handoff turn draws an edge to the *next turn actually produced by
    the named target agent* (not just the immediately-following turn - the
    target agent might not act until a few turns later, though in practice
    it's usually the very next agent turn). If no later turn belongs to
    that agent, the handoff still happened but has no destination turn to
    point at yet, so it's skipped rather than guessed at.
    """
    edges: list[GraphEdge] = []
    seen_agents: set[str] = set()
    for i, turn in enumerate(agent_turns):
        seen_agents.add(turn.agent)  # type: ignore[arg-type]
        if not turn.handoff_to:
            continue

        target_turn = next(
            (t for t in agent_turns[i + 1 :] if t.agent == turn.handoff_to), None
        )
        if target_turn is None:
            continue

        edge_failures = _merge_failures(
            failures_by_step.get(turn.step, []),
            failures_by_step.get(target_turn.step, []),
        )
        edges.append(
            GraphEdge(
                source=turn.agent,  # type: ignore[arg-type]
                target=turn.handoff_to,
                from_step=turn.step,
                to_step=target_turn.step,
                is_loopback=turn.handoff_to in seen_agents,
                explicit=True,
                failing=bool(edge_failures),
                failures=edge_failures,
            )
        )

    return edges


def _build_inferred_edges(
    agent_turns: list[Turn], failures_by_step: dict[int, list[FlaggedFailure]]
) -> list[GraphEdge]:
    """Build edges by inferring a handoff whenever `Turn.agent` changes
    between consecutive turns - the original strategy, used as a fallback
    for traces/frameworks with no structural handoff signal at all."""
    edges: list[GraphEdge] = []
    seen_agents: set[str] = set()
    for prev_turn, next_turn in zip(agent_turns, agent_turns[1:]):
        seen_agents.add(prev_turn.agent)  # type: ignore[arg-type]
        source, target = prev_turn.agent, next_turn.agent
        assert source is not None and target is not None
        if source == target:
            # Same agent producing consecutive turns (e.g. tool-call ->
            # follow-up) isn't a hand-off; skip the self-edge rather than
            # drawing a node pointing to itself.
            continue

        edge_failures = _merge_failures(
            failures_by_step.get(prev_turn.step, []),
            failures_by_step.get(next_turn.step, []),
        )
        edges.append(
            GraphEdge(
                source=source,
                target=target,
                from_step=prev_turn.step,
                to_step=next_turn.step,
                is_loopback=target in seen_agents,
                explicit=False,
                failing=bool(edge_failures),
                failures=edge_failures,
            )
        )

    return edges


def _merge_failures(
    existing: list[FlaggedFailure], new: list[FlaggedFailure]
) -> list[FlaggedFailure]:
    """Merge two failure lists, de-duplicating by identity (not equality) -
    the same FlaggedFailure object can legitimately be cited by both
    endpoints of an edge, but shouldn't be listed twice in one detail panel.
    """
    merged = list(existing)
    seen_ids = {id(f) for f in existing}
    for failure in new:
        if id(failure) not in seen_ids:
            merged.append(failure)
            seen_ids.add(id(failure))
    return merged


# --- Graph data for the client-side force layout ---------------------------
#
# Python computes structure and starting positions; the browser settles the
# final layout (see the `_JS` force simulation below) and draws the SVG
# itself. This keeps one graph description (this dict) as the single source
# of truth for both the visible page and the hover/pin detail content —
# there's no separate "rendered markup" and "detail data" to keep in sync.

# One hex color per MAST category, used to tint a fault node/edge/tooltip so
# failure *type* is visible before reading any text - amber for system-design
# (a build/config problem), magenta for inter-agent (a communication
# problem), red for verification (nothing checked the output, the most
# classically alarming category). Kept as a dict here (Python-side) and
# duplicated as a JS object literal below - both are static and small enough
# that generating one from the other would add indirection without benefit.
_CATEGORY_COLOR = {
    FailureCategory.SYSTEM_DESIGN: "#f5a623",
    FailureCategory.INTER_AGENT_MISALIGNMENT: "#e05fd6",
    FailureCategory.TASK_VERIFICATION: "#ff5a5f",
}


def _escape(text: str) -> str:
    return _html.escape(text, quote=True)


def _json_safe_for_inline_script(data: dict) -> str:
    r"""Serialize `data` to JSON safe to embed inside an inline `<script>`.

    Agent names ultimately come from a parsed trace file, so they're
    untrusted-ish input: an agent literally named e.g. `</script><script>...`
    must not be able to prematurely close the surrounding script tag and
    inject executable markup. `json.dumps` alone doesn't guard against
    this — it happily emits a literal `</script>` substring inside a JSON
    string value. Escaping the forward slash in that sequence (to `<\/`)
    is the standard fix: `\/` is a valid JSON escape for `/` per the JSON
    spec, so this changes nothing about how the data parses, only how the
    HTML tokenizer sees it.
    """
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def _dominant_category(failures: list[FlaggedFailure]) -> FailureCategory | None:
    """The category to color a node/edge by: the most frequently cited
    category among its failures, ties broken by category declaration order
    (System Design > Inter-Agent > Task Verification) for determinism."""
    if not failures:
        return None
    counts: dict[FailureCategory, int] = {}
    for f in failures:
        counts[f.category] = counts.get(f.category, 0) + 1
    ordered = list(FailureCategory)
    return max(counts, key=lambda c: (counts[c], -ordered.index(c)))


def _failure_detail_dict(failure: FlaggedFailure) -> dict[str, object]:
    mode_info = get_failure_mode(failure.failure_mode.value)
    return {
        "failure_mode": failure.failure_mode.value,
        "failure_mode_name": mode_info.name,
        "category": CATEGORY_NAMES[failure.category],
        "color": _CATEGORY_COLOR[failure.category],
        "justification": failure.justification,
        "confidence": failure.confidence,
        "turn_indices": list(failure.turn_indices),
    }


def _edge_interaction_summary(graph: Graph, edge: GraphEdge, trace: NormalizedTrace) -> str:
    """What actually happened at this handoff, for the edge's hover detail:
    the receiving turn's own message content, or a fallback describing tool
    activity if the turn had no text content of its own."""
    turns_by_step = {t.step: t for t in trace}
    target_turn = turns_by_step.get(edge.to_step)
    if target_turn is None:
        return f"{edge.source} handed off to {edge.target}."
    if target_turn.content:
        return target_turn.content
    if target_turn.tool_calls:
        names = ", ".join(c.name for c in target_turn.tool_calls)
        return f"{edge.target} invoked: {names}"
    return f"{edge.source} handed off to {edge.target} (no message content)."


def _node_radius(node: GraphNode, max_turns: int) -> float:
    """Radius scales with activity (turn count) so busier agents read as
    physically larger, not just differently colored."""
    if max_turns <= 1:
        return (_NODE_RADIUS_MIN + _NODE_RADIUS_MAX) / 2
    t = (len(node.turn_steps) - 1) / (max_turns - 1)
    return _NODE_RADIUS_MIN + t * (_NODE_RADIUS_MAX - _NODE_RADIUS_MIN)


def _initial_position(
    index: int, count: int, width: float, height: float
) -> tuple[float, float]:
    """A deterministic starting point on a circle - not the final layout
    (the client-side force simulation settles that), just a reasonable,
    non-overlapping seed so nodes don't all spawn stacked at the origin."""
    cx, cy = width / 2, height / 2
    if count <= 1:
        return cx, cy
    radius = min(width, height) * 0.32
    angle = (2 * math.pi * index / count) - math.pi / 2
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def _build_graph_payload(
    graph: Graph, trace: NormalizedTrace | None
) -> dict[str, object]:
    """The single JSON blob the client reads to build and animate the graph,
    and to answer "what's under the pointer" for hover/pin. Plain,
    JSON-serializable data - no pre-rendered markup - mirroring how
    `json_export.py` treats `ReportSummary` as data, not text."""
    max_turns = max((len(n.turn_steps) for n in graph.nodes), default=1)
    width, height = _canvas_size(len(graph.nodes))

    nodes = []
    for node in graph.nodes:
        x, y = _initial_position(node.index, len(graph.nodes), width, height)
        category = _dominant_category(node.failures)
        nodes.append(
            {
                "id": node.agent,
                "label": node.agent,
                "x": x,
                "y": y,
                "radius": _node_radius(node, max_turns),
                "turnCount": len(node.turn_steps),
                "turnSteps": list(node.turn_steps),
                "tools": list(node.tool_names),
                "failing": node.failing,
                "color": _CATEGORY_COLOR.get(category) if category else None,
                "failures": [_failure_detail_dict(f) for f in node.failures],
            }
        )

    edges = []
    for i, edge in enumerate(graph.edges):
        category = _dominant_category(edge.failures)
        interaction = (
            _edge_interaction_summary(graph, edge, trace) if trace is not None else ""
        )
        edges.append(
            {
                "id": f"edge-{i}",
                "source": edge.source,
                "target": edge.target,
                "loopback": edge.is_loopback,
                "explicit": edge.explicit,
                "failing": edge.failing,
                "color": _CATEGORY_COLOR.get(category) if category else None,
                "interaction": interaction,
                "failures": [_failure_detail_dict(f) for f in edge.failures],
            }
        )

    return {"nodes": nodes, "edges": edges, "width": width, "height": height}


# --- Full page assembly ------------------------------------------------------


def render_html(summary: ReportSummary, trace: NormalizedTrace | None = None) -> str:
    """Render a full, self-contained HTML report page.

    Args:
        summary: The report to render.
        trace: The source trace. Required to draw the graph (nodes/edges are
            derived from turn order); if omitted, the summary panel still
            renders but the graph section shows an explanatory placeholder
            instead of an empty canvas.
    """
    if trace is not None:
        graph = build_graph(trace, summary.flagged_failures)
    else:
        graph = Graph(nodes=[], edges=[])

    has_content = trace is not None and graph.nodes
    if trace is not None:
        graph_payload = _build_graph_payload(graph, trace)
    else:
        w, h = _canvas_size(0)
        graph_payload = {"nodes": [], "edges": [], "width": w, "height": h}
    payload_json = _json_safe_for_inline_script(graph_payload)

    empty_note = ""
    if trace is None:
        empty_note = (
            '<p class="empty-note">No trace attached to this report, so '
            "there's no graph to draw. Findings below still stand.</p>"
        )
    elif not graph.nodes:
        empty_note = '<p class="empty-note">No agent turns to visualize.</p>'

    legend_html = "".join(
        f'<span class="legend-item"><span class="legend-dot" '
        f'style="background:{color}"></span>{_escape(CATEGORY_NAMES[cat])}</span>'
        for cat, color in _CATEGORY_COLOR.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentDoc MAST Diagnosis</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <header class="masthead">
    <div class="masthead-mark" aria-hidden="true">
      <svg viewBox="0 0 28 16" class="mark-svg"><path d="M0 8 H8 L11 2 L15 14 L18 8 H28"
        class="mark-path"/></svg>
    </div>
    <div class="masthead-text">
      <p class="masthead-kicker">AGENTDOC &middot; TRACE ANALYSIS</p>
      <h1>Diagnosis Graph</h1>
    </div>
  </header>
  {_render_summary_section(summary)}
  <section class="graph-section">
    <div class="graph-section-head">
      <div>
        <h2>Agent Graph</h2>
        <p class="section-hint">Hover a node or connector for its finding. Click to pin it open.</p>
      </div>
      <div class="legend" aria-hidden="true">{legend_html}</div>
    </div>
    <div class="graph-wrapper" id="graph-wrapper">
      {empty_note}
      <div id="graph-tooltip" class="tooltip" hidden></div>
    </div>
  </section>
</main>
<script>
const AGENTDOC_GRAPH = {payload_json};
const AGENTDOC_HAS_CONTENT = {str(bool(has_content)).lower()};
{_JS}
</script>
</body>
</html>
"""


def _render_summary_section(summary: ReportSummary) -> str:
    ok = summary.total_failures == 0
    status_class = "summary-clean" if ok else "summary-fault"
    verdict = "CLEAN" if ok else "FAULT"

    fields = []
    if summary.source_framework:
        fields.append(("FRAMEWORK", _escape(summary.source_framework)))
    fields.append(("TURNS", str(summary.trace_turn_count)))
    if summary.model:
        fields.append(("MODEL", _escape(summary.model)))
    fields_html = "".join(
        f'<div class="readout-field"><span class="field-key">{key}</span>'
        f'<span class="field-val">{val}</span></div>'
        for key, val in fields
    )

    counts_html = ""
    top_modes_html = ""
    if summary.total_failures > 0:
        max_count = max((cc.count for cc in summary.category_counts), default=1) or 1
        bars = "".join(
            f"""<div class="count-row {'lit' if cc.count else ''}">
                  <span class="count-label">{_escape(CATEGORY_NAMES[cc.category])}</span>
                  <span class="count-bar-track">
                    <span class="count-bar-fill" style="width:{(cc.count / max_count) * 100:.0f}%;
                      background:{_CATEGORY_COLOR[cc.category] if cc.count else 'var(--ink-faint)'}"></span>
                  </span>
                  <span class="count-value">{cc.count}</span>
                </div>"""
            for cc in summary.category_counts
        )
        counts_html = f'<div class="category-counts">{bars}</div>'

        if summary.ranked_failure_modes:
            items = "".join(
                f"<li><span class=\"fm-code\">{_escape(fmc.failure_mode.value)}</span> "
                f"{_escape(get_failure_mode(fmc.failure_mode.value).name)} "
                f"<span class=\"fm-count\">&times;{fmc.count}</span></li>"
                for fmc in summary.ranked_failure_modes[:3]
            )
            top_modes_html = (
                f'<p class="top-modes-label">Recurring</p>'
                f'<ul class="top-modes">{items}</ul>'
            )

    return f"""
  <section class="summary-section {status_class}">
    <div class="readout-top">
      <span class="verdict-lamp" aria-hidden="true"></span>
      <span class="verdict-text">{verdict}</span>
      <div class="readout-fields">{fields_html}</div>
    </div>
    <p class="narrative">{_escape(summary.narrative)}</p>
    {counts_html}
    {top_modes_html}
  </section>
""".strip()


_CSS = """
:root {
  color-scheme: dark light;
  --bg: #0a0d11;
  --panel: #12171d;
  --panel-raised: #171d24;
  --grid: #1d242c;
  --hair: #263038;
  --ink: #d7dee5;
  --ink-dim: #7c8994;
  --ink-faint: #4c5761;
  --trace: #3ea6ff;
  --trace-dim: #285a80;
  --fault: #ff5a5f;
  --fault-dim: #7a2e30;
  --fault-glow: rgba(255, 90, 95, 0.28);
  --lamp-off: #2a323a;
}
@media (prefers-color-scheme: light) {
  :root:not([data-force-dark]) {
    --bg: #eef1f4;
    --panel: #ffffff;
    --panel-raised: #f7f9fb;
    --grid: #e3e8ec;
    --hair: #d3dae0;
    --ink: #1b232b;
    --ink-dim: #5b6771;
    --ink-faint: #94a0a9;
    --trace: #1266ad;
    --trace-dim: #a9cfe8;
    --fault: #c81e26;
    --fault-dim: #f2c8ca;
    --fault-glow: rgba(200, 30, 38, 0.16);
    --lamp-off: #d3dae0;
  }
}
* { box-sizing: border-box; }
html { color-scheme: dark light; }
body {
  margin: 0;
  padding: 2.5rem 1.5rem 4rem;
  background: var(--bg);
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 28px 28px;
  color: var(--ink);
  font-family: ui-sans-serif, "Segoe UI", system-ui, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
code, .mono { font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
main { max-width: 900px; margin: 0 auto; }

/* Masthead -------------------------------------------------------------- */
.masthead { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 1.6rem; }
.masthead-mark { flex: none; width: 44px; height: 26px; }
.mark-svg { width: 100%; height: 100%; }
.mark-path { fill: none; stroke: var(--trace); stroke-width: 1.6; stroke-linejoin: round; stroke-linecap: round; }
.masthead-kicker {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 0.68rem; letter-spacing: 0.14em; color: var(--ink-faint); margin: 0 0 0.2rem;
}
.masthead h1 { font-size: 1.3rem; font-weight: 650; margin: 0; letter-spacing: -0.01em; }

/* Sections ---------------------------------------------------------------*/
section {
  margin-bottom: 1.25rem;
  padding: 1.1rem 1.25rem;
  border-radius: 4px;
  border: 1px solid var(--hair);
  background: var(--panel);
}
h2 {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--ink-faint); margin: 0 0 0.9rem; font-weight: 600;
}
.section-hint { color: var(--ink-dim); font-size: 0.82rem; margin: -0.5rem 0 0.9rem; }

/* Summary readout --------------------------------------------------------*/
.summary-section { position: relative; overflow: hidden; }
.summary-section::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--trace);
}
.summary-fault::before { background: var(--fault); }
.readout-top { display: flex; align-items: center; flex-wrap: wrap; gap: 0.9rem 1.3rem; margin-bottom: 0.9rem; }
.verdict-lamp {
  width: 9px; height: 9px; border-radius: 50%; flex: none;
  background: var(--trace); box-shadow: 0 0 8px 1px var(--trace);
}
.summary-fault .verdict-lamp { background: var(--fault); box-shadow: 0 0 9px 2px var(--fault-glow); animation: pulse 2.2s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) { .summary-fault .verdict-lamp { animation: none; } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
.verdict-text {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08em; color: var(--ink);
}
.readout-fields { display: flex; flex-wrap: wrap; gap: 1.1rem; margin-left: auto; }
.readout-field { display: flex; flex-direction: column; gap: 0.1rem; }
.field-key {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 0.62rem; letter-spacing: 0.1em; color: var(--ink-faint);
}
.field-val {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 0.78rem; color: var(--ink-dim);
}
.narrative { margin: 0 0 0.9rem; font-size: 0.95rem; line-height: 1.5; max-width: 62ch; }

.category-counts { display: flex; flex-direction: column; gap: 0.45rem; margin-bottom: 0.9rem; }
.count-row { display: grid; grid-template-columns: 11rem 1fr 1.6rem; align-items: center; gap: 0.7rem; }
.count-label {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 0.72rem; color: var(--ink-faint);
}
.count-row.lit .count-label { color: var(--ink); }
.count-bar-track { height: 5px; border-radius: 2px; background: var(--grid); overflow: hidden; }
.count-bar-fill { display: block; height: 100%; background: var(--ink-faint); border-radius: 2px; }
.count-value {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 0.75rem; text-align: right; color: var(--ink-dim);
}
.count-row.lit .count-value { color: var(--ink); font-weight: 600; }

.top-modes-label {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 0.68rem; letter-spacing: 0.08em; color: var(--ink-faint); margin: 0.7rem 0 0.35rem;
}
.top-modes { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.85rem; }
.fm-code {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  color: var(--fault); font-size: 0.78rem; margin-right: 0.4rem;
}
.fm-count { color: var(--ink-faint); font-size: 0.8rem; }

/* Graph section head ---------------------------------------------------- */
.graph-section-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap; }
.legend { display: flex; gap: 0.9rem; flex-wrap: wrap; padding-top: 0.15rem; }
.legend-item {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 0.68rem; color: var(--ink-faint); display: inline-flex; align-items: center; gap: 0.4rem;
  white-space: nowrap;
}
.legend-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }

/* Graph canvas -----------------------------------------------------------*/
.graph-wrapper {
  position: relative; border-radius: 6px; background: var(--panel-raised);
  border: 1px solid var(--hair); overflow: hidden; touch-action: none;
}
svg.agentdoc-graph { display: block; width: 100%; height: auto; }
.empty-note { color: var(--ink-dim); margin: 0; padding: 2.5rem 1.25rem; font-size: 0.85rem; text-align: center; }

.gr-edge { fill: none; stroke: var(--ink-faint); stroke-width: 1; opacity: 0.4; transition: opacity 0.15s ease, stroke-width 0.15s ease; }
.gr-edge.fault { stroke-width: 2.25; opacity: 0.95; }
.gr-edge.dimmed { opacity: 0.12; }
.gr-edge.hot { opacity: 1; stroke-width: 3; }
/* An explicit (structurally-signaled) handoff: heavier stroke, higher base
   opacity so it doesn't recede like an inferred connection, and a directed
   arrowhead - the framework told us control passed this way, so it reads
   as a deliberate transfer rather than an ordinary tool-call badge. */
.gr-edge.handoff { stroke: var(--trace); stroke-width: 2; opacity: 0.75; }
.gr-edge.handoff.fault { stroke-width: 2.5; }
.gr-handoff-arrowhead { fill: var(--trace); }

.gr-node-halo { opacity: 0; transition: opacity 0.2s ease; }
.gr-node.fault .gr-node-halo { opacity: 0.55; }
.gr-node-circle {
  stroke-width: 1.6; transition: stroke-width 0.15s ease, filter 0.15s ease, opacity 0.15s ease;
  fill: var(--panel); stroke: var(--ink-faint); opacity: 0.55;
}
.gr-node.fault .gr-node-circle { opacity: 1; stroke-width: 2.25; }
.gr-node.dimmed .gr-node-circle { opacity: 0.18; }
.gr-node.hot .gr-node-circle { stroke-width: 3; }
.gr-node-label {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 11px; font-weight: 600; fill: var(--ink); text-anchor: middle; pointer-events: none;
  opacity: 0.85; transition: opacity 0.15s ease;
}
.gr-node.dimmed .gr-node-label { opacity: 0.15; }
.gr-node.fault .gr-node-label { opacity: 1; }
.gr-node { cursor: pointer; }
.gr-node:focus { outline: none; }
.gr-node:focus .gr-node-circle { stroke-width: 3; }
.gr-node.fault.pulse .gr-node-halo { animation: node-pulse 2.4s ease-in-out infinite; }
@keyframes node-pulse {
  0%, 100% { opacity: 0.35; r: var(--halo-r); }
  50% { opacity: 0.7; r: calc(var(--halo-r) + 5px); }
}
@media (prefers-reduced-motion: reduce) {
  .gr-node.fault.pulse .gr-node-halo { animation: none; opacity: 0.5; }
}

/* Floating tooltip ---------------------------------------------------------*/
.tooltip {
  position: absolute; z-index: 5; max-width: 300px; pointer-events: none;
  background: var(--panel); border: 1px solid var(--hair); border-radius: 6px;
  padding: 0.7rem 0.85rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);
  font-size: 0.82rem; line-height: 1.4; color: var(--ink);
  transition: opacity 0.1s ease;
}
.tooltip[hidden] { display: none; }
.tooltip.pinned { pointer-events: auto; }
.tooltip-title {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-weight: 700; font-size: 0.85rem; margin: 0 0 0.35rem; display: flex; align-items: center; gap: 0.4rem;
}
.tooltip-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.tooltip-meta {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  color: var(--ink-faint); font-size: 0.7rem; margin: 0 0 0.4rem;
}
.tooltip-clean { color: var(--ink-dim); margin: 0; }
.tooltip-fm { border-left: 2px solid var(--ink-faint); padding-left: 0.55rem; margin: 0.45rem 0 0; }
.tooltip-fm:first-of-type { margin-top: 0.5rem; }
.tooltip-fm-head { display: flex; align-items: baseline; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.15rem; }
.tooltip-fm-id {
  font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-weight: 700; font-size: 0.76rem;
}
.tooltip-fm-name { font-weight: 600; font-size: 0.82rem; }
.tooltip-fm-conf { color: var(--ink-faint); font-size: 0.7rem; }
.tooltip-fm-body { color: var(--ink); font-size: 0.8rem; margin: 0.15rem 0 0; }
.tooltip-pin-hint { color: var(--ink-faint); font-size: 0.68rem; margin: 0.5rem 0 0; font-style: italic; }
.tooltip-close {
  position: absolute; top: 0.4rem; right: 0.5rem; background: none; border: none;
  color: var(--ink-faint); font-size: 0.9rem; cursor: pointer; padding: 0.1rem 0.3rem; line-height: 1;
}
.tooltip-close:hover { color: var(--ink); }

@media (max-width: 560px) {
  body { padding: 1.5rem 1rem 3rem; }
  .readout-fields { margin-left: 0; }
  .count-row { grid-template-columns: 7.5rem 1fr 1.6rem; }
  .tooltip { max-width: 240px; }
}
"""

_JS = """
(function () {
  var graph = AGENTDOC_GRAPH;
  if (!AGENTDOC_HAS_CONTENT || !graph.nodes.length) return;

  var wrapper = document.getElementById('graph-wrapper');
  var tooltip = document.getElementById('graph-tooltip');
  var W = graph.width, H = graph.height;
  var reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var SVGNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  var svg = el('svg', {
    viewBox: '0 0 ' + W + ' ' + H,
    class: 'agentdoc-graph',
    role: 'img',
    'aria-label': 'Force-directed graph of ' + graph.nodes.length + ' agents'
  });

  var defs = el('defs', {});
  var glow = el('filter', { id: 'gr-glow', x: '-80%', y: '-80%', width: '260%', height: '260%' });
  var blur = el('feGaussianBlur', { stdDeviation: '4', result: 'blur' });
  var merge = el('feMerge', {});
  merge.appendChild(el('feMergeNode', { in: 'blur' }));
  merge.appendChild(el('feMergeNode', { in: 'SourceGraphic' }));
  glow.appendChild(blur);
  glow.appendChild(merge);
  defs.appendChild(glow);
  // Arrowhead used only on explicit (structurally-signaled) handoff edges -
  // inferred edges stay plain undirected lines, so a real "agent A handed
  // off to agent B" reads as visually distinct from a guessed transition,
  // not just another tool-call badge.
  var handoffArrow = el('marker', {
    id: 'gr-handoff-arrow', viewBox: '0 0 10 10', refX: '8', refY: '5',
    markerWidth: '7', markerHeight: '7', orient: 'auto-start-reverse'
  });
  handoffArrow.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', class: 'gr-handoff-arrowhead' }));
  defs.appendChild(handoffArrow);
  svg.appendChild(defs);

  var edgeLayer = el('g', { class: 'edge-layer' });
  var nodeLayer = el('g', { class: 'node-layer' });
  svg.appendChild(edgeLayer);
  svg.appendChild(nodeLayer);
  wrapper.insertBefore(svg, tooltip);

  var byId = {};
  graph.nodes.forEach(function (n) {
    n.vx = 0; n.vy = 0;
    byId[n.id] = n;
  });

  // ---- Force simulation ---------------------------------------------------
  // A minimal, well-known force layout: nodes repel each other (Coulomb-like,
  // inverse-square falloff), edges pull their endpoints together (a spring
  // toward a rest length), and everything is pulled gently toward the
  // canvas center so the graph doesn't drift off-screen. This runs for a
  // fixed number of steps and stops - it settles once, it isn't a live
  // physics toy running forever.
  var REPULSION = 2600;
  var SPRING_LENGTH = 150;
  var SPRING_STRENGTH = 0.02;
  var CENTER_PULL = 0.012;
  var DAMPING = 0.82;
  var MAX_STEPS = reduceMotion ? 0 : 220;

  function step() {
    var n = graph.nodes;
    for (var i = 0; i < n.length; i++) {
      var a = n[i];
      var fx = (W / 2 - a.x) * CENTER_PULL;
      var fy = (H / 2 - a.y) * CENTER_PULL;
      for (var j = 0; j < n.length; j++) {
        if (i === j) continue;
        var b = n[j];
        var dx = a.x - b.x, dy = a.y - b.y;
        var distSq = Math.max(dx * dx + dy * dy, 60);
        var dist = Math.sqrt(distSq);
        var force = REPULSION / distSq;
        fx += (dx / dist) * force;
        fy += (dy / dist) * force;
      }
      a.fx = fx; a.fy = fy;
    }
    graph.edges.forEach(function (e) {
      var s = byId[e.source], t = byId[e.target];
      var dx = t.x - s.x, dy = t.y - s.y;
      var dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      var stretch = (dist - SPRING_LENGTH) * SPRING_STRENGTH;
      var ux = dx / dist, uy = dy / dist;
      s.fx += ux * stretch; s.fy += uy * stretch;
      t.fx -= ux * stretch; t.fy -= uy * stretch;
    });
    var moving = false;
    n.forEach(function (a) {
      if (a.pinnedDrag) return;
      a.vx = (a.vx + a.fx) * DAMPING;
      a.vy = (a.vy + a.fy) * DAMPING;
      a.x += a.vx; a.y += a.vy;
      a.x = Math.max(a.radius + 8, Math.min(W - a.radius - 8, a.x));
      a.y = Math.max(a.radius + 8, Math.min(H - a.radius - 8, a.y));
      if (Math.abs(a.vx) + Math.abs(a.vy) > 0.05) moving = true;
    });
    return moving;
  }

  // ---- Build DOM elements ---------------------------------------------------
  var edgeEls = graph.edges.map(function (e) {
    var cls = 'gr-edge';
    if (e.explicit) cls += ' handoff';
    if (e.failing) cls += ' fault';
    var attrs = {
      class: cls,
      'data-id': e.id,
      tabindex: '0',
      role: 'img',
      'aria-label': (e.explicit ? 'Handoff from ' : '') + e.source + ' to ' + e.target
    };
    if (e.explicit) attrs['marker-end'] = 'url(#gr-handoff-arrow)';
    var path = el('path', attrs);
    if (e.failing && e.color) path.setAttribute('stroke', e.color);
    edgeLayer.appendChild(path);
    return { data: e, path: path };
  });

  var nodeEls = graph.nodes.map(function (n) {
    var g = el('g', {
      class: 'gr-node' + (n.failing ? ' fault pulse' : ''),
      tabindex: '0',
      role: 'img',
      'aria-label': n.label
    });
    var haloR = n.radius + 7;
    var halo = el('circle', { class: 'gr-node-halo', r: haloR, style: '--halo-r:' + haloR + 'px' });
    if (n.color) { halo.setAttribute('fill', n.color); }
    var circle = el('circle', { class: 'gr-node-circle', r: n.radius });
    if (n.failing && n.color) {
      circle.setAttribute('stroke', n.color);
      circle.setAttribute('filter', 'url(#gr-glow)');
    }
    var label = el('text', { class: 'gr-node-label', dy: '0.35em' });
    label.textContent = n.label;
    g.appendChild(halo);
    g.appendChild(circle);
    g.appendChild(label);
    nodeLayer.appendChild(g);
    return { data: n, g: g, halo: halo, circle: circle, label: label };
  });

  function place() {
    edgeEls.forEach(function (ee) {
      var s = byId[ee.data.source], t = byId[ee.data.target];
      var dx = t.x - s.x, dy = t.y - s.y;
      var dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      var ux = dx / dist, uy = dy / dist;
      var sx = s.x + ux * s.radius, sy = s.y + uy * s.radius;
      var tx = t.x - ux * t.radius, ty = t.y - uy * t.radius;
      var mx = (sx + tx) / 2 - uy * (ee.data.loopback ? 34 : 14);
      var my = (sy + ty) / 2 + ux * (ee.data.loopback ? 34 : 14);
      ee.path.setAttribute('d', 'M ' + sx + ' ' + sy + ' Q ' + mx + ' ' + my + ' ' + tx + ' ' + ty);
    });
    nodeEls.forEach(function (ne) {
      ne.g.setAttribute('transform', 'translate(' + ne.data.x + ',' + ne.data.y + ')');
    });
  }

  function tick() {
    var moving = step();
    place();
    if (moving) requestAnimationFrame(tick);
  }
  place();
  if (MAX_STEPS > 0) {
    var stepsLeft = MAX_STEPS;
    (function loop() {
      var moving = step();
      place();
      stepsLeft -= 1;
      if (moving && stepsLeft > 0) requestAnimationFrame(loop);
    })();
  }

  // ---- Hover-first tooltip, click-to-pin -------------------------------------
  var pinned = null;

  function fmBlock(f) {
    return '<div class="tooltip-fm">' +
      '<div class="tooltip-fm-head"><span class="tooltip-fm-id" style="color:' + f.color + '">' +
      escapeHtml(f.failure_mode) + '</span><span class="tooltip-fm-name">' +
      escapeHtml(f.failure_mode_name) + '</span></div>' +
      '<div class="tooltip-fm-conf">' + escapeHtml(f.category) + ' &middot; confidence ' +
      f.confidence.toFixed(2) + ' &middot; turns ' + f.turn_indices.join(', ') + '</div>' +
      '<p class="tooltip-fm-body">' + escapeHtml(f.justification) + '</p></div>';
  }

  function nodeTooltipHtml(n) {
    var dot = n.color || 'var(--ink-faint)';
    var html = '<div class="tooltip-title"><span class="tooltip-dot" style="background:' + dot + '"></span>' +
      escapeHtml(n.label) + '</div>' +
      '<div class="tooltip-meta">' + n.turnCount + ' turn' + (n.turnCount === 1 ? '' : 's') +
      (n.tools.length ? ' &middot; tools: ' + escapeHtml(n.tools.join(', ')) : '') + '</div>';
    if (!n.failing) {
      html += '<p class="tooltip-clean">No flagged failure on this agent.</p>';
    } else {
      n.failures.forEach(function (f) { html += fmBlock(f); });
    }
    return html;
  }

  function edgeTooltipHtml(e) {
    var dot = e.color || 'var(--ink-faint)';
    var html = '<div class="tooltip-title"><span class="tooltip-dot" style="background:' + dot + '"></span>' +
      escapeHtml(e.source) + ' &rarr; ' + escapeHtml(e.target) + '</div>';
    if (e.explicit) {
      html += '<div class="tooltip-meta">Explicit handoff (agent-declared)</div>';
    }
    if (e.interaction) {
      html += '<p class="tooltip-clean" style="color:var(--ink)">' + escapeHtml(e.interaction) + '</p>';
    }
    if (!e.failing) {
      html += '<p class="tooltip-clean">No flagged failure on this handoff.</p>';
    } else {
      e.failures.forEach(function (f) { html += fmBlock(f); });
    }
    return html;
  }

  function showTooltip(html, clientX, clientY, isPin) {
    tooltip.innerHTML = (isPin ? '<button class="tooltip-close" aria-label="Close">&times;</button>' : '') + html;
    tooltip.hidden = false;
    tooltip.classList.toggle('pinned', !!isPin);
    positionTooltip(clientX, clientY);
    if (isPin) {
      var closeBtn = tooltip.querySelector('.tooltip-close');
      if (closeBtn) closeBtn.addEventListener('click', function () { pinned = null; hideTooltip(); });
    }
  }

  function positionTooltip(clientX, clientY) {
    var rect = wrapper.getBoundingClientRect();
    var x = clientX - rect.left + 14;
    var y = clientY - rect.top + 14;
    if (x + 300 > rect.width) x = clientX - rect.left - 300 - 14;
    tooltip.style.left = Math.max(4, x) + 'px';
    tooltip.style.top = Math.max(4, y) + 'px';
  }

  function hideTooltip() {
    if (pinned) return;
    tooltip.hidden = true;
  }

  // Highlight the hovered/pinned node or edge, and dim everything not
  // directly connected to it, so attention narrows to one neighborhood
  // instead of the whole graph staying equally bright.
  function setHot(hotNode, hotEdge) {
    var neighborIds = null;
    if (hotNode) {
      neighborIds = {};
      neighborIds[hotNode.data.id] = true;
      graph.edges.forEach(function (e) {
        if (e.source === hotNode.data.id) neighborIds[e.target] = true;
        if (e.target === hotNode.data.id) neighborIds[e.source] = true;
      });
    }
    nodeEls.forEach(function (ne) {
      var isHot = ne === hotNode;
      ne.g.classList.toggle('hot', isHot);
      ne.g.classList.toggle('dimmed', !!neighborIds && !neighborIds[ne.data.id]);
    });
    edgeEls.forEach(function (ee) {
      var isHotEdge = ee === hotEdge;
      var touchesHotNode = !!hotNode &&
        (ee.data.source === hotNode.data.id || ee.data.target === hotNode.data.id);
      ee.path.classList.toggle('hot', isHotEdge || touchesHotNode);
      ee.path.classList.toggle(
        'dimmed',
        (!!hotNode && !touchesHotNode) || (!!hotEdge && !isHotEdge)
      );
    });
  }

  function clearHot() {
    nodeEls.forEach(function (ne) { ne.g.classList.remove('hot', 'dimmed'); });
    edgeEls.forEach(function (ee) { ee.path.classList.remove('hot', 'dimmed'); });
  }

  nodeEls.forEach(function (ne) {
    ne.g.addEventListener('mouseenter', function (ev) {
      if (pinned) return;
      setHot(ne);
      showTooltip(nodeTooltipHtml(ne.data), ev.clientX, ev.clientY, false);
    });
    ne.g.addEventListener('mousemove', function (ev) {
      if (pinned) return;
      positionTooltip(ev.clientX, ev.clientY);
    });
    ne.g.addEventListener('mouseleave', function () {
      if (pinned) return;
      clearHot();
      hideTooltip();
    });
    ne.g.addEventListener('click', function (ev) {
      pinned = ne.g;
      setHot(ne);
      showTooltip(nodeTooltipHtml(ne.data), ev.clientX, ev.clientY, true);
    });
    ne.g.addEventListener('focus', function () {
      var box = ne.g.getBoundingClientRect();
      setHot(ne);
      showTooltip(nodeTooltipHtml(ne.data), box.left + box.width / 2, box.top);
    });
    ne.g.addEventListener('blur', function () { if (!pinned) { clearHot(); hideTooltip(); } });
    ne.g.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        pinned = ne.g;
        var box = ne.g.getBoundingClientRect();
        showTooltip(nodeTooltipHtml(ne.data), box.left + box.width / 2, box.top, true);
      }
    });
  });

  edgeEls.forEach(function (ee) {
    ee.path.addEventListener('mouseenter', function (ev) {
      if (pinned) return;
      setHot(null, ee);
      showTooltip(edgeTooltipHtml(ee.data), ev.clientX, ev.clientY, false);
    });
    ee.path.addEventListener('mousemove', function (ev) {
      if (pinned) return;
      positionTooltip(ev.clientX, ev.clientY);
    });
    ee.path.addEventListener('mouseleave', function () {
      if (pinned) return;
      clearHot();
      hideTooltip();
    });
    ee.path.addEventListener('click', function (ev) {
      pinned = ee.path;
      setHot(null, ee);
      showTooltip(edgeTooltipHtml(ee.data), ev.clientX, ev.clientY, true);
    });
  });

  wrapper.addEventListener('click', function (ev) {
    if (ev.target === wrapper || ev.target === svg) {
      pinned = null;
      clearHot();
      hideTooltip();
    }
  });

  // ---- Drag to reposition (reinforces this is a live physics layout, not
  // a static picture). A plain click still pins the tooltip (see the click
  // handlers above); dragging only engages once the pointer actually moves
  // past a small threshold, so the two gestures don't fight over the same
  // mousedown. While dragging, the node is held at the pointer (pinnedDrag);
  // releasing hands it back to the simulation immediately, so it doesn't
  // stay frozen in place after the drag ends.
  var dragging = null;
  var dragMoved = false;
  var dragStart = null;
  var DRAG_THRESHOLD = 4;
  function svgPoint(clientX, clientY) {
    var rect = svg.getBoundingClientRect();
    return {
      x: (clientX - rect.left) * (W / rect.width),
      y: (clientY - rect.top) * (H / rect.height)
    };
  }
  nodeEls.forEach(function (ne) {
    ne.g.addEventListener('mousedown', function (ev) {
      dragging = ne;
      dragMoved = false;
      dragStart = { x: ev.clientX, y: ev.clientY };
    });
  });
  window.addEventListener('mousemove', function (ev) {
    if (!dragging) return;
    if (!dragMoved) {
      var moved = Math.abs(ev.clientX - dragStart.x) + Math.abs(ev.clientY - dragStart.y);
      if (moved < DRAG_THRESHOLD) return;
      dragMoved = true;
      dragging.data.pinnedDrag = true;
    }
    var p = svgPoint(ev.clientX, ev.clientY);
    dragging.data.x = p.x;
    dragging.data.y = p.y;
    dragging.data.vx = 0;
    dragging.data.vy = 0;
    place();
  });
  window.addEventListener('mouseup', function () {
    if (dragging) {
      dragging.data.pinnedDrag = false;
      requestAnimationFrame(function () {
        var stepsLeft = 60;
        (function loop() {
          var moving = step();
          place();
          stepsLeft -= 1;
          if (moving && stepsLeft > 0) requestAnimationFrame(loop);
        })();
      });
    }
    dragging = null;
  });
})();
"""


def write_report_html(
    summary: ReportSummary, path: Path, trace: NormalizedTrace | None = None
) -> None:
    """Write a full HTML report to `path` (UTF-8, creates/overwrites)."""
    Path(path).write_text(render_html(summary, trace), encoding="utf-8")
