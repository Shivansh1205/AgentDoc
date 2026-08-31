"""Parser for LangGraph execution traces.

Target format
-------------
This parser targets a "simplified stream capture": a JSON array of the
per-superstep chunks LangGraph emits from `graph.stream(...)`, e.g. via a
custom callback/logger that dumps each chunk as it's produced. Each element
looks like::

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

This mirrors two real LangGraph/LangChain conventions:

- LangGraph's `.stream()` yields one dict per superstep, keyed by node name,
  with the node's contribution to shared state (we accept a flattened
  `{"step", "node", "state", "timestamp"}` shape rather than the raw
  `{node_name: state}` mapping, since that's what most custom loggers
  actually persist).
- `state["messages"]` follows LangChain's `BaseMessage` JSON shape: `type` is
  one of "human" / "ai" / "system" / "tool"; an "ai" message may carry
  `tool_calls` (list of `{id, name, args}`); a "tool" message carries the
  `tool_call_id` and `name` of the call it answers.

Because LangGraph's shared state is cumulative (each superstep's
`state.messages` is the full running list, not just new messages), this
parser diffs against previously-seen messages so each message becomes exactly
one `Turn`, and folds a tool's result message back into the `ToolCall` that
requested it rather than emitting a redundant standalone turn.
"""

from __future__ import annotations

import json
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


class LangGraphParseError(ValueError):
    """Raised when a file does not match the expected LangGraph trace shape."""


class LangGraphParser(TraceParser):
    """Parses a LangGraph stream-capture JSON export into a `NormalizedTrace`."""

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
        seen_message_keys: set[tuple[str | None, str | None, str | None]] = set()
        turn_step = 0

        for chunk_index, chunk in enumerate(data):
            if not isinstance(chunk, dict):
                raise LangGraphParseError(
                    f"Chunk {chunk_index} is not a JSON object: {chunk!r}"
                )

            node = chunk.get("node")
            timestamp = chunk.get("timestamp")
            state = chunk.get("state") or {}
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


def _message_identity(
    message: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """A best-effort identity for de-duplicating cumulative message state.

    LangChain messages don't guarantee a stable id, so we fall back to
    (type, tool_call_id, content) which is stable enough for a single trace.
    """
    return (
        message.get("type"),
        message.get("tool_call_id"),
        message.get("content") if isinstance(message.get("content"), str) else None,
    )


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
    for raw_call in message.get("tool_calls") or []:
        call = ToolCall(
            name=raw_call.get("name", ""),
            call_id=raw_call.get("id"),
            args=raw_call.get("args") or {},
        )
        tool_calls.append(call)
        if call.call_id:
            pending_tool_calls[call.call_id] = call

    agent = message.get("name") or (default_agent if role is Role.AGENT else None)

    return Turn(
        step=step,
        role=role,
        agent=agent,
        content=message.get("content"),
        tool_calls=tool_calls,
        timestamp=timestamp,
        metadata={"chunk_index": chunk_index, "node": default_agent},
    )
