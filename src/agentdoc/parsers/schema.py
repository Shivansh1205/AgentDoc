"""Framework-agnostic normalized trace schema.

Every parser in this package (LangGraph, and future ones for AutoGen, CrewAI,
etc.) reads a framework's native trace format and produces a `NormalizedTrace`
made of `Turn`s. This is the single contract the MAST classifier depends on —
it should never need to know which framework produced a trace.

Design notes:
- A "turn" is one unit of activity attributable to a single role/agent at a
  single step (e.g. one node's output in a LangGraph superstep). A turn may
  itself contain multiple tool calls, since a single agent response often
  requests several tools before handing off.
- `step` is the ordinal position of the turn within the trace. It is the
  primary ordering signal parsers must fill in reliably, even when a
  framework's own timestamps are missing or unreliable.
- `parent_step` / `agent` give the classifier enough structure to reconstruct
  who was talking to whom without needing a full graph representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Who produced a turn's content."""

    SYSTEM = "system"
    HUMAN = "human"
    AGENT = "agent"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A single tool invocation requested by an agent, and its result if known."""

    name: str
    call_id: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None


@dataclass
class Turn:
    """One normalized unit of activity in a multi-agent trace.

    Attributes:
        step: Ordinal position of this turn in the trace (0-indexed).
        agent: Name of the agent/node/actor that produced this turn
            (e.g. "researcher", "writer"). None for turns with no clear
            owner (e.g. a raw system message).
        role: Coarse category of who produced the turn.
        content: The text content of the turn, if any.
        tool_calls: Tool calls made during this turn, with results attached
            when available.
        timestamp: ISO-8601 timestamp string, if the source trace provided one.
        parent_step: Step index this turn logically follows/responds to, when
            it differs from a simple `step - 1` (e.g. branching, handoffs).
            None when linear order is sufficient.
        metadata: Anything framework-specific worth preserving for debugging
            or future classifier features, without polluting the core schema.
    """

    step: int
    role: Role
    agent: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: str | None = None
    parent_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedTrace:
    """A full multi-agent execution trace, normalized to a sequence of turns.

    Attributes:
        turns: The ordered sequence of turns making up the trace.
        source_framework: Name of the framework the trace was parsed from
            (e.g. "langgraph"), for provenance and debugging.
        metadata: Trace-level metadata (e.g. run id, graph name) that doesn't
            belong to any single turn.
    """

    turns: list[Turn] = field(default_factory=list)
    source_framework: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self):
        return iter(self.turns)
