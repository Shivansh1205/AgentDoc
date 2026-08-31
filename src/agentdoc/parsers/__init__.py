"""Framework-specific trace parsers.

Each parser reads a framework's native trace/log format (e.g. LangGraph,
CrewAI, AutoGen) and converts it into the common `NormalizedTrace` format
defined in `agentdoc.parsers.schema`, so the MAST classifier never needs to
know which framework produced a given trace.

Adding a new framework: create `<framework>_parser.py`, implement a
`TraceParser` subclass (see `agentdoc.parsers.base`), and register it in
`PARSERS` below.
"""

from agentdoc.parsers.base import TraceParser
from agentdoc.parsers.langgraph_parser import LangGraphParser
from agentdoc.parsers.schema import NormalizedTrace, Role, ToolCall, Turn

#: Registry of available parsers, keyed by framework identifier.
PARSERS: dict[str, type[TraceParser]] = {
    LangGraphParser.framework: LangGraphParser,
}

__all__ = [
    "PARSERS",
    "NormalizedTrace",
    "Role",
    "ToolCall",
    "Turn",
    "TraceParser",
    "LangGraphParser",
]
