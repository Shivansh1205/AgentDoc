"""Tests for the LangGraph trace parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdoc.parsers.langgraph_parser import LangGraphParseError, LangGraphParser
from agentdoc.parsers.schema import Role

EXAMPLE_TRACE = Path(__file__).parent.parent / "examples" / "langgraph_trace_example.json"
FLAWED_EXAMPLE_TRACE = (
    Path(__file__).parent.parent / "examples" / "langgraph_trace_flawed_example.json"
)


@pytest.fixture
def parser() -> LangGraphParser:
    return LangGraphParser()


def test_parses_example_trace_end_to_end(parser: LangGraphParser) -> None:
    trace = parser.parse(EXAMPLE_TRACE)

    assert trace.source_framework == "langgraph"
    # system, human, ai(+tool_call), ai(final researcher), ai(writer) = 5 turns.
    # The tool result is folded into the researcher's tool call, not its own turn.
    assert len(trace) == 5


def test_turns_are_sequential_and_zero_indexed(parser: LangGraphParser) -> None:
    trace = parser.parse(EXAMPLE_TRACE)
    assert [turn.step for turn in trace] == list(range(len(trace)))


def test_system_and_human_turns(parser: LangGraphParser) -> None:
    trace = parser.parse(EXAMPLE_TRACE)

    assert trace.turns[0].role is Role.SYSTEM
    assert "two-agent research team" in trace.turns[0].content

    assert trace.turns[1].role is Role.HUMAN
    assert "GDP growth rate" in trace.turns[1].content


def test_tool_call_and_result_are_folded_into_one_turn(parser: LangGraphParser) -> None:
    trace = parser.parse(EXAMPLE_TRACE)

    tool_call_turn = trace.turns[2]
    assert tool_call_turn.role is Role.AGENT
    assert tool_call_turn.agent == "researcher"
    assert len(tool_call_turn.tool_calls) == 1

    call = tool_call_turn.tool_calls[0]
    assert call.name == "web_search"
    assert call.call_id == "call_1"
    assert call.args == {"query": "US GDP growth rate latest quarter 2026"}
    assert "2.1%" in call.result

    # No standalone "tool" role turn should exist for a resolved call.
    assert all(turn.role is not Role.TOOL for turn in trace)


def test_agent_handoff_reflected_in_turns(parser: LangGraphParser) -> None:
    trace = parser.parse(EXAMPLE_TRACE)

    researcher_turns = [t for t in trace if t.agent == "researcher"]
    writer_turns = [t for t in trace if t.agent == "writer"]

    assert len(researcher_turns) == 2
    assert len(writer_turns) == 1
    assert "2.1%" in writer_turns[0].content


def test_cumulative_state_does_not_duplicate_turns(parser: LangGraphParser) -> None:
    # Each chunk in the example re-sends the full cumulative message list;
    # the parser must not re-emit messages already seen in an earlier chunk.
    trace = parser.parse(EXAMPLE_TRACE)
    contents = [t.content for t in trace if t.content]
    assert len(contents) == len(set(contents))


def test_timestamps_are_preserved(parser: LangGraphParser) -> None:
    trace = parser.parse(EXAMPLE_TRACE)
    assert trace.turns[0].timestamp == "2026-08-27T10:00:00Z"


def test_rejects_non_list_json(tmp_path: Path, parser: LangGraphParser) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(LangGraphParseError, match="JSON array"):
        parser.parse(bad_file)


def test_rejects_invalid_json(tmp_path: Path, parser: LangGraphParser) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(LangGraphParseError, match="not valid JSON"):
        parser.parse(bad_file)


def test_rejects_missing_file(tmp_path: Path, parser: LangGraphParser) -> None:
    with pytest.raises(LangGraphParseError, match="Could not read"):
        parser.parse(tmp_path / "does_not_exist.json")


def test_rejects_unrecognized_message_type(tmp_path: Path, parser: LangGraphParser) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "node": "researcher",
                    "state": {"messages": [{"type": "mystery", "content": "??"}]},
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(LangGraphParseError, match="Unrecognized message type"):
        parser.parse(bad_file)


def test_unresolved_tool_call_still_surfaces_result_as_turn(
    tmp_path: Path, parser: LangGraphParser
) -> None:
    # A tool message with no matching prior tool_call (e.g. truncated trace)
    # should still surface as its own turn rather than being silently dropped.
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "node": "researcher",
                    "state": {
                        "messages": [
                            {
                                "type": "tool",
                                "tool_call_id": "orphan_call",
                                "name": "web_search",
                                "content": "orphaned result",
                            }
                        ]
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    trace = parser.parse(trace_file)
    assert len(trace) == 1
    assert trace.turns[0].role is Role.TOOL
    assert trace.turns[0].content == "orphaned result"


# ---------------------------------------------------------------------------
# Flawed regression fixture (examples/langgraph_trace_flawed_example.json).
# This only checks that the fixture parses into the shape the classifier
# regression check relies on; see examples/README.md for what MAST failures
# it's designed to exercise and how to run that check.
# ---------------------------------------------------------------------------


def test_flawed_example_parses_end_to_end(parser: LangGraphParser) -> None:
    trace = parser.parse(FLAWED_EXAMPLE_TRACE)

    assert trace.source_framework == "langgraph"
    assert len(trace) == 7


def test_flawed_example_contains_the_duplicate_tool_call(parser: LangGraphParser) -> None:
    # The injected step-repetition failure (FM-1.3): two web_search calls
    # with the identical query and identical result.
    trace = parser.parse(FLAWED_EXAMPLE_TRACE)

    search_calls = [
        call
        for turn in trace
        for call in turn.tool_calls
        if call.name == "web_search"
    ]
    assert len(search_calls) == 2
    assert search_calls[0].args == search_calls[1].args
    assert search_calls[0].result == search_calls[1].result


def test_flawed_example_contains_the_contradicting_writer_turn(
    parser: LangGraphParser,
) -> None:
    # The injected inter-agent-misalignment failure (FM-2.5): the writer's
    # figure must actually differ from the researcher's confirmed figure for
    # this fixture to mean anything.
    trace = parser.parse(FLAWED_EXAMPLE_TRACE)

    researcher_turns = [t for t in trace if t.agent == "researcher"]
    writer_turns = [t for t in trace if t.agent == "writer"]
    assert researcher_turns and writer_turns

    assert "412 million" in researcher_turns[-1].content
    assert "380 million" in writer_turns[0].content


def test_flawed_example_contains_unconditional_completion_turn(
    parser: LangGraphParser,
) -> None:
    # The injected task-verification failure (FM-3.1 / FM-3.2): a supervisor
    # turn that declares completion without checking anything.
    trace = parser.parse(FLAWED_EXAMPLE_TRACE)

    supervisor_turns = [t for t in trace if t.agent == "supervisor"]
    assert len(supervisor_turns) == 1
    assert "complete" in supervisor_turns[0].content.lower()
