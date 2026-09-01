"""Tests for `agentdoc.report.html` (self-contained HTML graph visualization).

Covers two layers: `build_graph()` (pure data - nodes/edges derived from a
trace + flagged failures, independent of any HTML/SVG string generation) and
`render_html()` (the full page). No test here needs a browser - "valid
enough to not obviously break" is checked via a balanced-tag scan and by
confirming the expected content strings actually appear in the output,
mirroring how `test_report_terminal.py` checks Rich output without needing a
real terminal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agentdoc.classifier.results import ClassificationResult, FlaggedFailure
from agentdoc.classifier.taxonomy import FailureCategory, FailureMode
from agentdoc.parsers.langgraph_parser import LangGraphParser
from agentdoc.parsers.schema import NormalizedTrace, Role, ToolCall, Turn
from agentdoc.report.generator import generate_report
from agentdoc.report.html import build_graph, render_html

SWARM_EXAMPLE_TRACE = (
    Path(__file__).parent.parent / "examples" / "langgraph_swarm_example.json"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _extract_graph_payload(html: str) -> dict:
    """Pull the `AGENTDOC_GRAPH` JSON blob back out of a rendered page, the
    single source of truth the client-side force layout and hover tooltips
    are built from - asserting against this is more robust than matching
    exact generated markup, since the DOM is now constructed by JS rather
    than emitted as static HTML/SVG strings."""
    match = re.search(r"const AGENTDOC_GRAPH = (\{.*?\});\n", html, re.DOTALL)
    assert match, "AGENTDOC_GRAPH payload not found in rendered HTML"
    return json.loads(match.group(1))


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
                content="Let me check.",
                tool_calls=[ToolCall(name="lookup_spec", call_id="c1")],
            ),
            Turn(
                step=2,
                role=Role.AGENT,
                agent="writer",
                content="The spec is version 1.0.",
            ),
        ],
    )


@pytest.fixture
def revisit_trace() -> NormalizedTrace:
    """researcher -> writer -> researcher again: exercises loopback edges
    and multi-turn-step node merging."""
    return NormalizedTrace(
        source_framework="langgraph",
        turns=[
            Turn(step=0, role=Role.HUMAN, content="Question"),
            Turn(step=1, role=Role.AGENT, agent="researcher", content="Looking..."),
            Turn(step=2, role=Role.AGENT, agent="writer", content="Draft answer."),
            Turn(step=3, role=Role.AGENT, agent="researcher", content="Correction needed."),
            Turn(step=4, role=Role.AGENT, agent="writer", content="Final answer."),
        ],
    )


@pytest.fixture
def explicit_handoff_trace() -> NormalizedTrace:
    """A swarm-style trace where the same agent makes several tool calls
    (search, then book, then transfer) before a handoff, mirroring the real
    langgraph-swarm-py shape: only the *last* flight_assistant turn should
    produce a handoff edge, not the earlier ones that also made tool calls
    in the same logical burst."""
    return NormalizedTrace(
        source_framework="langgraph",
        turns=[
            Turn(step=0, role=Role.HUMAN, content="Book my trip"),
            Turn(
                step=1,
                role=Role.AGENT,
                agent="flight_assistant",
                tool_calls=[ToolCall(name="search_flights", call_id="c1")],
            ),
            Turn(
                step=2,
                role=Role.AGENT,
                agent="flight_assistant",
                tool_calls=[ToolCall(name="book_flight", call_id="c2")],
            ),
            Turn(
                step=3,
                role=Role.AGENT,
                agent="flight_assistant",
                tool_calls=[ToolCall(name="transfer_to_hotel_assistant", call_id="c3")],
                handoff_to="hotel_assistant",
            ),
            Turn(
                step=4,
                role=Role.AGENT,
                agent="hotel_assistant",
                content="Booked your hotel too.",
            ),
        ],
    )


def _failure(
    mode: FailureMode,
    category: FailureCategory,
    turns: list[int],
    justification: str = "Something went wrong here.",
    confidence: float = 0.85,
) -> FlaggedFailure:
    return FlaggedFailure(
        failure_mode=mode,
        category=category,
        turn_indices=turns,
        justification=justification,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# build_graph: node/edge derivation
# ---------------------------------------------------------------------------


def test_build_graph_one_node_per_distinct_agent(sample_trace: NormalizedTrace) -> None:
    graph = build_graph(sample_trace, [])
    agents = [n.agent for n in graph.nodes]
    assert agents == ["researcher", "writer"]


def test_build_graph_skips_turns_with_no_agent(sample_trace: NormalizedTrace) -> None:
    # The human turn (step 0, no agent) must not become a node.
    graph = build_graph(sample_trace, [])
    assert all(n.agent for n in graph.nodes)
    assert len(graph.nodes) == 2


def test_build_graph_node_order_matches_first_appearance(sample_trace: NormalizedTrace) -> None:
    graph = build_graph(sample_trace, [])
    assert graph.nodes[0].index == 0
    assert graph.nodes[1].index == 1


def test_build_graph_edge_sequence(sample_trace: NormalizedTrace) -> None:
    graph = build_graph(sample_trace, [])
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "researcher"
    assert graph.edges[0].target == "writer"
    assert graph.edges[0].is_loopback is False
    # sample_trace has no Turn.handoff_to signals, so this edge must come
    # from the inferred (consecutive-turns-differ) strategy, not the
    # explicit one.
    assert graph.edges[0].explicit is False


def test_build_graph_tool_calls_become_node_badges_not_nodes(
    sample_trace: NormalizedTrace,
) -> None:
    graph = build_graph(sample_trace, [])
    researcher = next(n for n in graph.nodes if n.agent == "researcher")
    assert researcher.tool_names == ["lookup_spec"]
    # Tool calls must not inflate node count - still exactly 2 agent nodes.
    assert len(graph.nodes) == 2


def test_build_graph_revisit_reuses_same_node(revisit_trace: NormalizedTrace) -> None:
    graph = build_graph(revisit_trace, [])
    # Exactly one node per distinct agent, even though researcher appears
    # in two non-consecutive turns.
    assert len(graph.nodes) == 2
    researcher = next(n for n in graph.nodes if n.agent == "researcher")
    assert researcher.turn_steps == [1, 3]


def test_build_graph_revisit_produces_loopback_edge(revisit_trace: NormalizedTrace) -> None:
    graph = build_graph(revisit_trace, [])
    # researcher -> writer -> researcher -> writer: the second and third
    # edges revisit an already-seen agent.
    assert len(graph.edges) == 3
    assert graph.edges[0].is_loopback is False  # researcher -> writer (first time)
    assert graph.edges[1].is_loopback is True  # writer -> researcher (revisit)
    assert graph.edges[2].is_loopback is True  # researcher -> writer (revisit)


# ---------------------------------------------------------------------------
# Explicit handoff edges (Turn.handoff_to), vs. the inferred fallback.
# ---------------------------------------------------------------------------


def test_build_graph_uses_explicit_handoff_when_present(
    explicit_handoff_trace: NormalizedTrace,
) -> None:
    graph = build_graph(explicit_handoff_trace, [])

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.source == "flight_assistant"
    assert edge.target == "hotel_assistant"
    assert edge.explicit is True
    assert edge.from_step == 3  # the turn with the handoff_to, not step 1 or 2
    assert edge.to_step == 4


def test_build_graph_explicit_handoff_ignores_earlier_same_agent_tool_calls(
    explicit_handoff_trace: NormalizedTrace,
) -> None:
    # flight_assistant makes three tool calls across steps 1-3, but only
    # step 3 (the one with handoff_to set) should produce an edge - the
    # inferred strategy would also skip 1->2 (same agent) but this confirms
    # the *explicit* strategy does too, and specifically anchors the edge to
    # the turn that actually declared the handoff.
    graph = build_graph(explicit_handoff_trace, [])
    assert len(graph.edges) == 1


def test_build_graph_falls_back_to_inference_when_no_handoff_to_present(
    sample_trace: NormalizedTrace,
) -> None:
    # sample_trace's turns have no handoff_to at all - confirms the
    # fallback path (not just that it happens to produce the same answer).
    graph = build_graph(sample_trace, [])
    assert len(graph.edges) == 1
    assert graph.edges[0].explicit is False


def test_build_graph_explicit_handoff_with_no_matching_target_turn_is_skipped() -> None:
    # A handoff_to naming an agent that never actually produces a later
    # turn (e.g. a truncated trace) has no destination to point at - it
    # should be dropped, not guessed at.
    trace = NormalizedTrace(
        turns=[
            Turn(
                step=0,
                role=Role.AGENT,
                agent="flight_assistant",
                handoff_to="hotel_assistant",
            ),
        ]
    )
    graph = build_graph(trace, [])
    assert graph.edges == []
    # The node for flight_assistant should still exist even though its
    # handoff went nowhere.
    assert len(graph.nodes) == 1


def test_build_graph_explicit_handoff_failure_attribution(
    explicit_handoff_trace: NormalizedTrace,
) -> None:
    failure = _failure(
        FailureMode.IGNORED_OTHER_AGENTS_INPUT,
        FailureCategory.INTER_AGENT_MISALIGNMENT,
        turns=[3, 4],
    )
    graph = build_graph(explicit_handoff_trace, [failure])

    assert len(graph.edges) == 1
    assert graph.edges[0].failing is True
    assert graph.edges[0].failures == [failure]


def test_build_graph_consecutive_same_agent_turns_produce_no_self_edge() -> None:
    trace = NormalizedTrace(
        turns=[
            Turn(step=0, role=Role.AGENT, agent="researcher", content="a"),
            Turn(step=1, role=Role.AGENT, agent="researcher", content="b"),
            Turn(step=2, role=Role.AGENT, agent="writer", content="c"),
        ]
    )
    graph = build_graph(trace, [])
    assert len(graph.nodes) == 2
    # Only one edge (researcher -> writer); no self-loop for step0->step1.
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "researcher"


def test_build_graph_empty_trace_produces_empty_graph() -> None:
    graph = build_graph(NormalizedTrace(turns=[]), [])
    assert graph.nodes == []
    assert graph.edges == []


# ---------------------------------------------------------------------------
# build_graph: failure attribution
# ---------------------------------------------------------------------------


def test_build_graph_marks_node_failing_when_its_turn_is_cited(
    sample_trace: NormalizedTrace,
) -> None:
    failure = _failure(
        FailureMode.REASONING_ACTION_MISMATCH,
        FailureCategory.INTER_AGENT_MISALIGNMENT,
        turns=[1],
    )
    graph = build_graph(sample_trace, [failure])

    researcher = next(n for n in graph.nodes if n.agent == "researcher")
    writer = next(n for n in graph.nodes if n.agent == "writer")
    assert researcher.failing is True
    assert writer.failing is False
    assert researcher.failures == [failure]


def test_build_graph_marks_edge_failing_when_either_endpoint_cited(
    sample_trace: NormalizedTrace,
) -> None:
    failure = _failure(
        FailureMode.NO_OR_INCOMPLETE_VERIFICATION,
        FailureCategory.TASK_VERIFICATION,
        turns=[2],  # writer's turn only
    )
    graph = build_graph(sample_trace, [failure])

    assert len(graph.edges) == 1
    assert graph.edges[0].failing is True
    assert graph.edges[0].failures == [failure]


def test_build_graph_healthy_nodes_have_no_failures(sample_trace: NormalizedTrace) -> None:
    graph = build_graph(sample_trace, [])
    assert all(n.failing is False for n in graph.nodes)
    assert all(n.failures == [] for n in graph.nodes)
    assert all(e.failing is False for e in graph.edges)


def test_build_graph_failure_spanning_multiple_agents_marks_both(
    revisit_trace: NormalizedTrace,
) -> None:
    # A single failure citing turns from both researcher and writer must
    # mark both nodes failing, and be attributed without duplication.
    failure = _failure(
        FailureMode.IGNORED_OTHER_AGENTS_INPUT,
        FailureCategory.INTER_AGENT_MISALIGNMENT,
        turns=[1, 2],
    )
    graph = build_graph(revisit_trace, [failure])

    researcher = next(n for n in graph.nodes if n.agent == "researcher")
    writer = next(n for n in graph.nodes if n.agent == "writer")
    assert researcher.failing and writer.failing
    assert researcher.failures == [failure]
    assert writer.failures == [failure]


def test_build_graph_does_not_duplicate_failure_on_merged_node(
    revisit_trace: NormalizedTrace,
) -> None:
    # A failure citing both of researcher's turns (1 and 3) should appear
    # exactly once in that node's failure list, not twice.
    failure = _failure(
        FailureMode.STEP_REPETITION,
        FailureCategory.SYSTEM_DESIGN,
        turns=[1, 3],
    )
    graph = build_graph(revisit_trace, [failure])
    researcher = next(n for n in graph.nodes if n.agent == "researcher")
    assert researcher.failures == [failure]


# ---------------------------------------------------------------------------
# render_html: structural sanity ("valid enough to not obviously break")
# ---------------------------------------------------------------------------


def _assert_tags_balanced(html: str) -> None:
    """A lightweight structural check: every opening tag from a small set of
    structurally-important elements has a matching closing tag count. Not a
    full HTML validator, but enough to catch a broken f-string/template
    that leaves a dangling <div> or duplicated </html>."""
    for tag in ["html", "head", "body", "svg", "script", "style", "main", "section"]:
        opens = len(re.findall(rf"<{tag}[ >]", html))
        closes = len(re.findall(rf"</{tag}>", html))
        assert opens == closes, f"<{tag}> open/close mismatch: {opens} vs {closes}"


def test_render_html_zero_failures_renders_without_error(
    sample_trace: NormalizedTrace,
) -> None:
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(sample_trace, result)

    html = render_html(summary, sample_trace)

    assert "<!doctype html>" in html.lower()
    _assert_tags_balanced(html)
    assert "researcher" in html
    assert "writer" in html
    assert "no issues were flagged" in html.lower() or "no mast failure modes" in html.lower()


def test_render_html_includes_agent_names(sample_trace: NormalizedTrace) -> None:
    failure = _failure(
        FailureMode.REASONING_ACTION_MISMATCH,
        FailureCategory.INTER_AGENT_MISALIGNMENT,
        turns=[1],
    )
    result = ClassificationResult(flagged_failures=[failure], model="test-model")
    summary = generate_report(sample_trace, result)

    html = render_html(summary, sample_trace)

    assert "researcher" in html
    assert "writer" in html


def test_render_html_includes_failure_mode_names_and_ids(
    sample_trace: NormalizedTrace,
) -> None:
    failure = _failure(
        FailureMode.NO_OR_INCOMPLETE_VERIFICATION,
        FailureCategory.TASK_VERIFICATION,
        turns=[2],
        justification="Nobody checked the final answer against the source.",
    )
    result = ClassificationResult(flagged_failures=[failure], model="test-model")
    summary = generate_report(sample_trace, result)

    html = render_html(summary, sample_trace)

    assert "FM-3.2" in html
    assert "No or incomplete verification" in html
    assert "Nobody checked the final answer against the source." in html
    assert "Task Verification" in html
    _assert_tags_balanced(html)


def test_render_html_failing_node_is_marked_in_graph_payload(
    sample_trace: NormalizedTrace,
) -> None:
    failure = _failure(
        FailureMode.REASONING_ACTION_MISMATCH,
        FailureCategory.INTER_AGENT_MISALIGNMENT,
        turns=[1],
    )
    result = ClassificationResult(flagged_failures=[failure], model="test-model")
    summary = generate_report(sample_trace, result)

    html = render_html(summary, sample_trace)
    payload = _extract_graph_payload(html)

    researcher = next(n for n in payload["nodes"] if n["id"] == "researcher")
    writer = next(n for n in payload["nodes"] if n["id"] == "writer")
    assert researcher["failing"] is True
    assert researcher["color"] is not None
    assert writer["failing"] is False
    assert writer["color"] is None


def test_render_html_healthy_trace_has_no_failing_nodes_or_edges(
    sample_trace: NormalizedTrace,
) -> None:
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(sample_trace, result)

    html = render_html(summary, sample_trace)
    payload = _extract_graph_payload(html)

    assert all(not n["failing"] for n in payload["nodes"])
    assert all(not e["failing"] for e in payload["edges"])


def test_render_html_embeds_graph_payload_for_all_nodes_and_edges(
    sample_trace: NormalizedTrace,
) -> None:
    failure = _failure(
        FailureMode.REASONING_ACTION_MISMATCH,
        FailureCategory.INTER_AGENT_MISALIGNMENT,
        turns=[1],
        confidence=0.72,
    )
    result = ClassificationResult(flagged_failures=[failure], model="test-model")
    summary = generate_report(sample_trace, result)

    html = render_html(summary, sample_trace)
    payload = _extract_graph_payload(html)

    node_ids = {n["id"] for n in payload["nodes"]}
    assert node_ids == {"researcher", "writer"}
    assert len(payload["edges"]) == 1
    assert payload["edges"][0]["id"] == "edge-0"
    assert "0.72" in html


def test_render_html_interaction_script_present(sample_trace: NormalizedTrace) -> None:
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(sample_trace, result)
    html = render_html(summary, sample_trace)

    # Hover-first interaction with click-to-pin, plus the force simulation.
    assert "addEventListener" in html
    assert "graph-tooltip" in html
    assert "mouseenter" in html
    assert "function step()" in html  # the force-simulation step function


def test_render_html_without_trace_shows_placeholder_not_error() -> None:
    trace = NormalizedTrace(
        source_framework="langgraph",
        turns=[Turn(step=0, role=Role.AGENT, agent="researcher", content="hi")],
    )
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(trace, result)

    html = render_html(summary, trace=None)

    _assert_tags_balanced(html)
    assert "no graph to draw" in html


def test_render_html_empty_trace_shows_empty_graph_placeholder() -> None:
    trace = NormalizedTrace(turns=[])
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(trace, result)

    html = render_html(summary, trace)

    _assert_tags_balanced(html)
    assert "No agent turns to visualize" in html


def test_render_html_summary_matches_terminal_narrative_data(
    sample_trace: NormalizedTrace,
) -> None:
    # The HTML report's narrative/counts must be the same data the terminal
    # report shows - no separate/diverging computation.
    failures = [
        _failure(FailureMode.STEP_REPETITION, FailureCategory.SYSTEM_DESIGN, [1]),
        _failure(
            FailureMode.NO_OR_INCOMPLETE_VERIFICATION,
            FailureCategory.TASK_VERIFICATION,
            [2],
        ),
    ]
    result = ClassificationResult(flagged_failures=failures, model="test-model")
    summary = generate_report(sample_trace, result)

    html = render_html(summary, sample_trace)

    assert summary.narrative in html
    # Category names and their counts are present, even though they're
    # rendered as separate label/bar/value elements rather than one
    # combined "Label: N" text run.
    assert "System Design Issues" in html
    assert "Task Verification" in html
    assert "Inter-Agent Misalignment" in html
    assert '<span class="count-value">1</span>' in html
    assert '<span class="count-value">0</span>' in html


def test_render_html_unicode_heavy_justification_renders_cleanly(
    sample_trace: NormalizedTrace,
) -> None:
    # The same class of content that broke the legacy-console terminal
    # renderer (smart quotes, non-breaking hyphen, em dash, emoji, CJK).
    # HTML is UTF-8, so this must render as real characters, not crash or
    # mangle into replacement characters the way the terminal path did.
    unicode_justification = (
        "The agent's response contains a non‑breaking hyphen "
        "and “smart quotes” plus an em dash — here, "
        "as well as emoji \U0001f6a8 and CJK 日本語."
    )
    failure = _failure(
        FailureMode.REASONING_ACTION_MISMATCH,
        FailureCategory.INTER_AGENT_MISALIGNMENT,
        turns=[1],
        justification=unicode_justification,
    )
    result = ClassificationResult(flagged_failures=[failure], model="test-model")
    summary = generate_report(sample_trace, result)

    html = render_html(summary, sample_trace)

    _assert_tags_balanced(html)
    assert unicode_justification in html


def test_render_html_escapes_agent_names_against_injection() -> None:
    # Agent names come from parsed trace data (ultimately from the source
    # framework's trace file), so an agent literally named with a script tag
    # must not be able to break out of the inline <script> block the graph
    # payload is embedded in and inject executable markup. The payload is
    # JSON (built client-side into text via .textContent, never innerHTML
    # for untrusted strings), so the guarantee here is specifically that the
    # literal `</script>` sequence never appears unescaped in the page.
    trace = NormalizedTrace(
        turns=[
            Turn(step=0, role=Role.AGENT, agent="</script><script>alert(1)</script>", content="hi"),
        ]
    )
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(trace, result)

    html = render_html(summary, trace)

    assert "</script><script>alert(1)</script>" not in html
    payload = _extract_graph_payload(html)
    assert payload["nodes"][0]["id"] == "</script><script>alert(1)</script>"


def test_render_html_revisit_trace_renders_loopback_without_error(
    revisit_trace: NormalizedTrace,
) -> None:
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(revisit_trace, result)

    html = render_html(summary, revisit_trace)
    payload = _extract_graph_payload(html)

    _assert_tags_balanced(html)
    # Exactly one researcher node and one writer node despite 2 turns each.
    node_ids = [n["id"] for n in payload["nodes"]]
    assert node_ids.count("researcher") == 1
    assert node_ids.count("writer") == 1
    # The second researcher<->writer edge is a loopback (revisiting writer,
    # which already appeared before this edge's source turn).
    assert any(e["loopback"] for e in payload["edges"])


# ---------------------------------------------------------------------------
# Real external fixture: examples/langgraph_swarm_example.json - see
# tests/test_langgraph_parser.py for the parser-level assertions on this
# same file; these cover what the HTML graph does with it once parsed.
# ---------------------------------------------------------------------------


def test_render_html_swarm_example_renders_without_error() -> None:
    parser = LangGraphParser()
    trace = parser.parse(SWARM_EXAMPLE_TRACE)
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(trace, result)

    html = render_html(summary, trace)

    _assert_tags_balanced(html)
    assert "flight_assistant" in html
    assert "hotel_assistant" in html


def test_render_html_swarm_example_handoff_edge_is_explicit() -> None:
    parser = LangGraphParser()
    trace = parser.parse(SWARM_EXAMPLE_TRACE)
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(trace, result)

    html = render_html(summary, trace)
    payload = _extract_graph_payload(html)

    assert len(payload["edges"]) == 1
    edge = payload["edges"][0]
    assert edge["source"] == "flight_assistant"
    assert edge["target"] == "hotel_assistant"
    assert edge["explicit"] is True


def test_render_html_swarm_example_explicit_edge_gets_handoff_arrow_marker() -> None:
    # The explicit-handoff visual treatment (arrowhead marker + "handoff"
    # CSS class) must actually be wired into the rendered page's script,
    # not just present in the payload data.
    parser = LangGraphParser()
    trace = parser.parse(SWARM_EXAMPLE_TRACE)
    result = ClassificationResult(flagged_failures=[], model="test-model")
    summary = generate_report(trace, result)

    html = render_html(summary, trace)

    assert "gr-handoff-arrow" in html
    assert "'explicit'" not in html  # sanity: not accidentally stringified oddly
    assert "e.explicit" in html  # the JS branches on the payload's explicit flag
