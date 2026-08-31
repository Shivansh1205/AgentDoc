"""Tests for MAST classifier prompt construction and trace chunking."""

from __future__ import annotations

from agentdoc.classifier.prompts import (
    build_taxonomy_reference,
    build_user_prompt,
    chunk_trace,
    render_trace,
    render_turn,
)
from agentdoc.parsers.schema import NormalizedTrace, Role, ToolCall, Turn


def test_taxonomy_reference_mentions_all_14_modes() -> None:
    reference = build_taxonomy_reference()
    for fm_id in [
        "FM-1.1", "FM-1.2", "FM-1.3", "FM-1.4", "FM-1.5",
        "FM-2.1", "FM-2.2", "FM-2.3", "FM-2.4", "FM-2.5", "FM-2.6",
        "FM-3.1", "FM-3.2", "FM-3.3",
    ]:
        assert fm_id in reference


def test_render_turn_includes_tool_call_and_result() -> None:
    turn = Turn(
        step=1,
        role=Role.AGENT,
        agent="researcher",
        content="Checking the docs.",
        tool_calls=[
            ToolCall(name="lookup", call_id="c1", args={"q": "x"}, result="found x")
        ],
    )
    rendered = render_turn(turn)

    assert "[step 1]" in rendered
    assert "researcher" in rendered
    assert "lookup" in rendered
    assert "found x" in rendered


def test_render_trace_preserves_turn_order() -> None:
    trace = NormalizedTrace(
        turns=[
            Turn(step=0, role=Role.HUMAN, content="first"),
            Turn(step=1, role=Role.AGENT, agent="a", content="second"),
        ]
    )
    rendered = render_trace(trace)
    assert rendered.index("first") < rendered.index("second")


def test_build_user_prompt_includes_taxonomy_and_transcript() -> None:
    trace = NormalizedTrace(turns=[Turn(step=0, role=Role.HUMAN, content="hello there")])
    prompt = build_user_prompt(trace)

    assert "MAST Failure Mode Definitions" in prompt
    assert "hello there" in prompt
    assert "FM-1.1" in prompt


def test_chunk_trace_returns_single_chunk_for_short_trace() -> None:
    trace = NormalizedTrace(
        turns=[Turn(step=i, role=Role.HUMAN, content="short") for i in range(5)]
    )
    chunks = chunk_trace(trace)
    assert len(chunks) == 1
    assert chunks[0] is trace


def test_chunk_trace_splits_long_trace() -> None:
    long_content = "x" * 1000
    trace = NormalizedTrace(
        turns=[
            Turn(step=i, role=Role.HUMAN, content=long_content) for i in range(100)
        ]
    )
    chunks = chunk_trace(trace, char_budget=5000, overlap_turns=1)

    assert len(chunks) > 1
    # Every turn must appear in at least one chunk (no silently dropped turns).
    covered_steps = set()
    for chunk in chunks:
        covered_steps.update(t.step for t in chunk.turns)
    assert covered_steps == {t.step for t in trace.turns}


def test_chunk_trace_chunks_overlap() -> None:
    long_content = "x" * 1000
    trace = NormalizedTrace(
        turns=[
            Turn(step=i, role=Role.HUMAN, content=long_content) for i in range(20)
        ]
    )
    chunks = chunk_trace(trace, char_budget=3000, overlap_turns=2)

    assert len(chunks) > 1
    first_chunk_steps = {t.step for t in chunks[0].turns}
    second_chunk_steps = {t.step for t in chunks[1].turns}
    assert first_chunk_steps & second_chunk_steps  # non-empty overlap


def test_chunk_trace_handles_empty_trace() -> None:
    trace = NormalizedTrace(turns=[])
    chunks = chunk_trace(trace)
    assert chunks == [trace]
