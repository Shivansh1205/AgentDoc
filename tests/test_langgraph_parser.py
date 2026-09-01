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
SWARM_EXAMPLE_TRACE = (
    Path(__file__).parent.parent / "examples" / "langgraph_swarm_example.json"
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


# ---------------------------------------------------------------------------
# Envelope auto-detection: the flattened "simplified stream capture" shape
# and LangGraph's own raw `.stream(stream_mode="updates")` shape.
#
# The raw shape used to silently produce zero turns (both `chunk.get("node")`
# and `chunk.get("state")` are None on it, so `{}.get("messages")` quietly
# returns `[]`). These tests pin down that it now parses correctly, and that
# a chunk matching neither shape raises rather than being treated as empty.
# ---------------------------------------------------------------------------


def test_parses_raw_stream_envelope(parser: LangGraphParser, tmp_path: Path) -> None:
    # {"<node_name>": {"messages": [...]}} - what graph.stream() actually
    # yields with no logging middleware reshaping it, as opposed to this
    # parser's own flattened {"step", "node", "state", "timestamp"} shape.
    trace_file = tmp_path / "raw_stream.json"
    trace_file.write_text(
        json.dumps(
            [
                {
                    "researcher": {
                        "messages": [
                            {"type": "human", "content": "Find the GDP figure"},
                            {
                                "type": "ai",
                                "name": "researcher",
                                "content": "Found it: 2.1%",
                            },
                        ]
                    }
                }
            ]
        ),
        encoding="utf-8",
    )

    trace = parser.parse(trace_file)

    assert len(trace) == 2
    assert trace.turns[0].role is Role.HUMAN
    assert trace.turns[1].agent == "researcher"
    assert trace.turns[1].content == "Found it: 2.1%"


def test_raw_stream_envelope_does_not_silently_produce_zero_turns(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    # Regression guard for the exact historical bug: this shape must never
    # again parse to an empty trace with no error.
    trace_file = tmp_path / "raw_stream.json"
    trace_file.write_text(
        json.dumps(
            [{"flight_assistant": {"messages": [{"type": "human", "content": "hi"}]}}]
        ),
        encoding="utf-8",
    )

    trace = parser.parse(trace_file)

    assert len(trace) != 0
    assert len(trace) == 1


def test_flattened_and_raw_envelopes_can_be_mixed_across_chunks(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    # Not a realistic trace, but confirms detection is per-chunk, not a
    # whole-file mode switch - each chunk stands on its own.
    trace_file = tmp_path / "mixed.json"
    trace_file.write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "node": "researcher",
                    "state": {"messages": [{"type": "human", "content": "hi"}]},
                },
                {
                    "hotel_assistant": {
                        "messages": [
                            {"type": "human", "content": "hi"},
                            {"type": "ai", "name": "hotel_assistant", "content": "ok"},
                        ]
                    }
                },
            ]
        ),
        encoding="utf-8",
    )

    trace = parser.parse(trace_file)

    assert len(trace) == 2
    assert trace.turns[1].agent == "hotel_assistant"


def test_unrecognized_chunk_shape_raises_rather_than_parsing_as_empty(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    # Neither the flattened shape (no "node"/"state" keys) nor the raw
    # shape (more than one top-level key, or a value with no "messages").
    trace_file = tmp_path / "bad.json"
    trace_file.write_text(
        json.dumps([{"some_key": "not a state dict", "another_key": 123}]),
        encoding="utf-8",
    )

    with pytest.raises(LangGraphParseError, match="matches neither known LangGraph trace shape"):
        parser.parse(trace_file)


def test_single_key_chunk_without_messages_raises(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    # Single top-level key (looks like the raw shape at a glance) but its
    # value has no "messages" key - must not be silently treated as a
    # zero-message chunk.
    trace_file = tmp_path / "bad.json"
    trace_file.write_text(
        json.dumps([{"researcher": {"active_agent": "writer"}}]),
        encoding="utf-8",
    )

    with pytest.raises(LangGraphParseError, match="matches neither known LangGraph trace shape"):
        parser.parse(trace_file)


def test_flattened_envelope_with_non_dict_state_raises(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    trace_file = tmp_path / "bad.json"
    trace_file.write_text(
        json.dumps([{"step": 0, "node": "researcher", "state": "not a dict"}]),
        encoding="utf-8",
    )

    with pytest.raises(LangGraphParseError, match="'state'.*must be an object"):
        parser.parse(trace_file)


# ---------------------------------------------------------------------------
# Handoff detection: Turn.handoff_to populated from a
# `transfer_to_<agent_name>`-style tool call (the langgraph-swarm
# convention), independent of any `active_agent` state key.
# ---------------------------------------------------------------------------


def test_handoff_to_populated_from_transfer_tool_call(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    trace_file = tmp_path / "handoff.json"
    trace_file.write_text(
        json.dumps(
            [
                {
                    "flight_assistant": {
                        "messages": [
                            {"type": "human", "content": "book my trip"},
                            {
                                "type": "ai",
                                "name": "flight_assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "name": "transfer_to_hotel_assistant",
                                        "args": {},
                                    }
                                ],
                            },
                        ],
                        "active_agent": "hotel_assistant",
                    }
                }
            ]
        ),
        encoding="utf-8",
    )

    trace = parser.parse(trace_file)

    handoff_turn = next(t for t in trace if t.agent == "flight_assistant")
    assert handoff_turn.handoff_to == "hotel_assistant"


def test_handoff_to_none_when_no_transfer_tool_call(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    trace_file = tmp_path / "no_handoff.json"
    trace_file.write_text(
        json.dumps(
            [
                {
                    "researcher": {
                        "messages": [
                            {"type": "human", "content": "look this up"},
                            {
                                "type": "ai",
                                "name": "researcher",
                                "content": "",
                                "tool_calls": [
                                    {"id": "call_1", "name": "web_search", "args": {}}
                                ],
                            },
                        ]
                    }
                }
            ]
        ),
        encoding="utf-8",
    )

    trace = parser.parse(trace_file)

    ai_turn = next(t for t in trace if t.agent == "researcher")
    assert ai_turn.handoff_to is None


def test_handoff_to_does_not_fire_on_active_agent_alone(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    # A chunk's active_agent describes state *after the whole chunk* and
    # must not be applied to every AI message in it - only a message whose
    # own tool call matches the transfer_to_<agent> convention is a handoff.
    trace_file = tmp_path / "active_agent_only.json"
    trace_file.write_text(
        json.dumps(
            [
                {
                    "flight_assistant": {
                        "messages": [
                            {"type": "human", "content": "book my flight"},
                            {
                                "type": "ai",
                                "name": "flight_assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "name": "search_flights",
                                        "args": {},
                                    }
                                ],
                            },
                        ],
                        "active_agent": "hotel_assistant",
                    }
                }
            ]
        ),
        encoding="utf-8",
    )

    trace = parser.parse(trace_file)

    search_turn = next(t for t in trace if t.agent == "flight_assistant")
    assert search_turn.handoff_to is None


def test_handoff_to_never_names_the_turns_own_agent(
    parser: LangGraphParser, tmp_path: Path
) -> None:
    # A pathological transfer_to_<self> tool call shouldn't be treated as a
    # handoff - an agent can't hand off to itself.
    trace_file = tmp_path / "self_transfer.json"
    trace_file.write_text(
        json.dumps(
            [
                {
                    "researcher": {
                        "messages": [
                            {"type": "human", "content": "hi"},
                            {
                                "type": "ai",
                                "name": "researcher",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "name": "transfer_to_researcher",
                                        "args": {},
                                    }
                                ],
                            },
                        ]
                    }
                }
            ]
        ),
        encoding="utf-8",
    )

    trace = parser.parse(trace_file)

    turn = next(t for t in trace if t.agent == "researcher")
    assert turn.handoff_to is None


# ---------------------------------------------------------------------------
# Real external fixture: examples/langgraph_swarm_example.json, an
# unmodified capture from langgraph-swarm-py's customer_support example
# (see examples/README.md for the full provenance/what-it-tests writeup).
# This is what actually drove both the envelope-detection and handoff_to
# work above - a regression here means real-world compatibility broke.
# ---------------------------------------------------------------------------


def test_swarm_example_parses_to_expected_turn_count(parser: LangGraphParser) -> None:
    trace = parser.parse(SWARM_EXAMPLE_TRACE)
    assert trace.source_framework == "langgraph"
    assert len(trace) == 7


def test_swarm_example_agent_sequence(parser: LangGraphParser) -> None:
    trace = parser.parse(SWARM_EXAMPLE_TRACE)
    agents = [t.agent for t in trace]
    assert agents == [
        None,  # the initial human message
        "flight_assistant",
        "flight_assistant",
        "flight_assistant",
        "hotel_assistant",
        "hotel_assistant",
        "hotel_assistant",
    ]


def test_swarm_example_tool_calls_folded_correctly(parser: LangGraphParser) -> None:
    trace = parser.parse(SWARM_EXAMPLE_TRACE)
    flight_turns = [t for t in trace if t.agent == "flight_assistant"]
    hotel_turns = [t for t in trace if t.agent == "hotel_assistant"]

    flight_tool_names = [c.name for t in flight_turns for c in t.tool_calls]
    hotel_tool_names = [c.name for t in hotel_turns for c in t.tool_calls]

    assert flight_tool_names == [
        "search_flights",
        "book_flight",
        "transfer_to_hotel_assistant",
    ]
    assert hotel_tool_names == ["search_hotels", "book_hotel"]

    # No standalone "tool" role turns should remain - every ToolMessage in
    # the real capture answers a call made earlier in the same trace.
    assert all(t.role is not Role.TOOL for t in trace)

    # Tool call results actually resolved (not left as None).
    search_flights_call = next(
        c for t in flight_turns for c in t.tool_calls if c.name == "search_flights"
    )
    assert search_flights_call.result is not None
    assert "Jet Blue" in search_flights_call.result


def test_swarm_example_handoff_to_populated_on_transfer_turn(
    parser: LangGraphParser,
) -> None:
    trace = parser.parse(SWARM_EXAMPLE_TRACE)

    handoff_turns = [t for t in trace if t.handoff_to is not None]
    assert len(handoff_turns) == 1
    assert handoff_turns[0].agent == "flight_assistant"
    assert handoff_turns[0].handoff_to == "hotel_assistant"
    # It's specifically the turn with the transfer tool call, not an
    # earlier/later flight_assistant turn.
    assert any(
        c.name == "transfer_to_hotel_assistant" for c in handoff_turns[0].tool_calls
    )

    # Every other turn must NOT claim a handoff, including the other two
    # flight_assistant turns that also made tool calls (search/book) in the
    # same underlying chunk as the handoff.
    non_handoff_turns = [t for t in trace if t.handoff_to is None]
    assert len(non_handoff_turns) == 6


def test_swarm_example_final_answer_content(parser: LangGraphParser) -> None:
    trace = parser.parse(SWARM_EXAMPLE_TRACE)
    final_turn = trace.turns[-1]
    assert final_turn.agent == "hotel_assistant"
    assert "BOS" in final_turn.content
    assert "McKittrick" in final_turn.content
