"""Parser for LangGraph execution traces.

Target formats
--------------
This parser accepts two JSON shapes, auto-detected per chunk:

1. A "simplified stream capture" flattened envelope, e.g. via a custom
   callback/logger that dumps each superstep as it's produced::

    {
        "step": 0,
        "node": "researcher",
        "state": {
            "messages": [
                {"type": "human", "content": "Find the latest GDP figures"},
                {"type": "ai", "name": "researcher", "content": "Searching...",
                 "tool_calls": [{"id": "call_1", "name": "web_search",
                                 "args": {"query": "GDP 2026"}}]},
                {"type": "tool", "tool_call_id": "call_1", "name": "web_search",
                 "content": "US GDP grew 2.1%..."}
            ]
        },
        "timestamp": "2026-08-27T10:00:01Z"
    }

2. LangGraph's own raw `.stream(stream_mode="updates")` output: a dict with
   exactly one key, the node name, whose value is that node's state update::

    {
        "researcher": {
            "messages": [ ... same message shape as above ... ]
        }
    }

   This is what `graph.stream(...)` actually yields with no logging
   middleware in between — confirmed against a real captured trace from
   `langgraph-swarm-py` (see examples/langgraph_swarm_example.json). Earlier
   versions of this parser only accepted shape 1 and silently produced zero
   turns on shape 2, since `chunk.get("node")`/`chunk.get("state")` are both
   `None` on it and `{}.get("messages")` quietly returns `[]`. A chunk
   matching neither shape now raises `LangGraphParseError` instead of being
   treated as empty.

Either way, `state["messages"]` follows LangChain's `BaseMessage` JSON shape:
`type` is one of "human" / "ai" / "system" / "tool"; an "ai" message may
carry `tool_calls` (list of `{id, name, args}`); a "tool" message carries the
`tool_call_id` and `name` of the call it answers.

Because LangGraph's shared state is cumulative (each superstep's
`state.messages` is the full running list, not just new messages), this
parser diffs against previously-seen messages so each message becomes exactly
one `Turn`, and folds a tool's result message back into the `ToolCall` that
requested it rather than emitting a redundant standalone turn.

Handoff detection
------------------
`langgraph-swarm` (and similar swarm-style multi-agent libraries) represent
an agent handing off control as a specific tool call - conventionally named
`transfer_to_<agent_name>` by `create_handoff_tool` - whose result updates an
`active_agent` state key rather than being a normal domain action. This
parser populates `Turn.handoff_to` when a message's own tool calls match
that `transfer_to_<agent_name>` naming pattern.

The chunk-level `active_agent` state key (also part of the swarm library's
state schema) is deliberately *not* consulted for this: it describes state
*after the whole chunk*, and a single chunk can carry several AI messages
(e.g. search a flight, book it, then transfer) - every message in that
chunk would look equally "responsible" for the resulting active_agent
value, so only a per-message signal (the tool-call name) can correctly
attribute the handoff to the one message that actually performed it.

`handoff_to` is left `None` when no matching tool call is found - callers
(e.g. `agentdoc.report.html.build_graph`) fall back to inferring handoffs
from `Turn.agent` changing between consecutive turns, exactly as before
this concept existed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentdoc.parsers.base import TraceParser
from agentdoc.parsers.schema import NormalizedTrace, Role, ToolCall, Turn

_ROLE_BY_MESSAGE_TYPE = {
    "human": Role.HUMAN,
    "system": Role.SYSTEM,
    "ai": Role.AGENT,
    "tool": Role.TOOL,
}

# Matches the tool name `create_handoff_tool` generates by default in
# langgraph-swarm: `transfer_to_<agent_name>` (agent name normalized to
# snake_case). See the "Handoff detection" section of the module docstring
# for why this per-message tool-call name is the signal used, rather than
# the chunk-level `active_agent` state key.
_HANDOFF_TOOL_NAME_RE = re.compile(r"^transfer_to_(?P<agent>.+)$")


class LangGraphParseError(ValueError):
    """Raised when a file does not match a known LangGraph trace shape."""


class LangGraphParser(TraceParser):
    """Parses a LangGraph trace export (flattened or raw stream) into a `NormalizedTrace`."""

    framework = "langgraph"

    def parse(self, path: Path) -> NormalizedTrace:
        try:
            raw_text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise LangGraphParseError(f"Could not read trace file: {path}") from exc

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LangGraphParseError(f"Trace file is not valid JSON: {path}") from exc

        if not isinstance(data, list):
            raise LangGraphParseError(
                "Expected a JSON array of superstep chunks "
                f"(got {type(data).__name__})."
            )

        trace = NormalizedTrace(source_framework=self.framework)
        # Tool-call results resolve against calls made earlier in the trace,
        # tracked by call_id across the whole run (not just within one step).
        pending_tool_calls: dict[str, ToolCall] = {}
        seen_message_keys: set[tuple[str | None, str | None, str | None, tuple[str, ...]]] = set()
        turn_step = 0

        for chunk_index, chunk in enumerate(data):
            if not isinstance(chunk, dict):
                raise LangGraphParseError(
                    f"Chunk {chunk_index} is not a JSON object: {chunk!r}"
                )

            node, state, timestamp = _unwrap_chunk(chunk, chunk_index)
            messages = state.get("messages") or []

            if not isinstance(messages, list):
                raise LangGraphParseError(
                    f"Chunk {chunk_index} ('state.messages') must be a list, "
                    f"got {type(messages).__name__}."
                )

            for message in messages:
                key = _message_identity(message)
                if key in seen_message_keys:
                    # Cumulative state already surfaced this message in an
                    # earlier chunk; skip so each message yields one Turn.
                    continue
                seen_message_keys.add(key)

                turn = _message_to_turn(
                    message,
                    step=turn_step,
                    default_agent=node,
                    timestamp=timestamp,
                    chunk_index=chunk_index,
                    pending_tool_calls=pending_tool_calls,
                )
                if turn is not None:
                    trace.turns.append(turn)
                    turn_step += 1

        return trace


def _unwrap_chunk(
    chunk: dict[str, Any], chunk_index: int
) -> tuple[str | None, dict[str, Any], str | None]:
    """Detect and unwrap one of the two accepted chunk envelopes.

    Returns (node_name, state_dict, timestamp). Raises `LangGraphParseError`
    if `chunk` matches neither known shape, rather than treating it as an
    empty chunk with no messages.
    """
    has_flattened_keys = "node" in chunk and "state" in chunk
    if has_flattened_keys:
        state = chunk.get("state")
        if not isinstance(state, dict):
            raise LangGraphParseError(
                f"Chunk {chunk_index} ('state') must be an object, "
                f"got {type(state).__name__}."
            )
        return chunk.get("node"), state, chunk.get("timestamp")

    # Raw LangGraph `.stream(stream_mode="updates")` shape: exactly one key,
    # the node name, whose value is that node's state update dict.
    if len(chunk) == 1:
        (node_name, state), = chunk.items()
        if isinstance(state, dict) and "messages" in state:
            return node_name, state, None

    raise LangGraphParseError(
        f"Chunk {chunk_index} matches neither known LangGraph trace shape: "
        "expected either {'node', 'state', ...} (flattened stream capture) "
        "or {'<node_name>': {'messages': [...], ...}} (raw .stream() "
        f"output). Got keys: {sorted(chunk.keys())!r}"
    )


def _message_identity(
    message: dict[str, Any],
) -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
    """A best-effort identity for de-duplicating cumulative message state.

    LangChain messages don't guarantee a stable id, so we fall back to
    (type, tool_call_id, content, tool_call_ids) - stable enough for a
    single trace. The tool-call-ids component matters for agent turns that
    make a tool call with no accompanying text: an agent that calls several
    tools back-to-back (e.g. search, then book, then hand off) typically
    produces multiple AI messages that all have `content=""` - real
    behavior seen from Groq/langgraph-swarm, not a hypothetical - so
    `(type, tool_call_id, content)` alone would collide across all of them
    and only the first would survive dedup. Tool call ids are unique per
    call, so including them (sorted, for order-independence) distinguishes
    these messages correctly.
    """
    tool_call_ids = tuple(
        sorted(
            call.get("id", "")
            for call in (message.get("tool_calls") or [])
            if call.get("id")
        )
    )
    return (
        message.get("type"),
        message.get("tool_call_id"),
        message.get("content") if isinstance(message.get("content"), str) else None,
        tool_call_ids,
    )


def _handoff_target_from_tool_calls(raw_tool_calls: list[dict[str, Any]]) -> str | None:
    """The agent name from the first `transfer_to_<agent_name>`-style tool
    call on this message, or None if none of its tool calls look like a
    handoff."""
    for raw_call in raw_tool_calls:
        match = _HANDOFF_TOOL_NAME_RE.match(raw_call.get("name") or "")
        if match:
            return match.group("agent")
    return None


def _message_to_turn(
    message: dict[str, Any],
    *,
    step: int,
    default_agent: str | None,
    timestamp: str | None,
    chunk_index: int,
    pending_tool_calls: dict[str, ToolCall],
) -> Turn | None:
    """Convert one LangChain-style message dict into a `Turn`, or None to skip."""
    message_type = message.get("type")
    role = _ROLE_BY_MESSAGE_TYPE.get(message_type)
    if role is None:
        raise LangGraphParseError(
            f"Unrecognized message type {message_type!r} in chunk {chunk_index}."
        )

    if role is Role.TOOL:
        # Fold the tool's result into the ToolCall that requested it instead
        # of emitting a standalone turn, so a tool round-trip reads as one
        # unit on the requesting agent's turn.
        call_id = message.get("tool_call_id")
        call = pending_tool_calls.get(call_id) if call_id else None
        if call is not None:
            call.result = message.get("content")
            return None
        # No matching call found (e.g. truncated trace) — surface it as its
        # own turn rather than silently dropping information.
        return Turn(
            step=step,
            role=Role.TOOL,
            agent=message.get("name") or default_agent,
            content=message.get("content"),
            timestamp=timestamp,
            metadata={"tool_call_id": call_id, "chunk_index": chunk_index},
        )

    tool_calls = []
    raw_tool_calls = message.get("tool_calls") or []
    for raw_call in raw_tool_calls:
        call = ToolCall(
            name=raw_call.get("name", ""),
            call_id=raw_call.get("id"),
            args=raw_call.get("args") or {},
        )
        tool_calls.append(call)
        if call.call_id:
            pending_tool_calls[call.call_id] = call

    agent = message.get("name") or (default_agent if role is Role.AGENT else None)

    # A message only actually performs a handoff if *this message's own*
    # tool calls include a transfer_to_<agent> pattern - see the module
    # docstring's "Handoff detection" section for why the chunk-level
    # `active_agent` state key isn't used for this instead.
    handoff_to: str | None = None
    if role is Role.AGENT:
        candidate = _handoff_target_from_tool_calls(raw_tool_calls)
        if candidate and candidate != agent:
            handoff_to = candidate

    return Turn(
        step=step,
        role=role,
        agent=agent,
        content=message.get("content"),
        tool_calls=tool_calls,
        timestamp=timestamp,
        handoff_to=handoff_to,
        metadata={"chunk_index": chunk_index, "node": default_agent},
    )
