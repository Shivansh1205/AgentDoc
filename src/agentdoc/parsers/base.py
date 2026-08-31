"""Base interface that every framework-specific parser implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agentdoc.parsers.schema import NormalizedTrace


class TraceParser(ABC):
    """Common interface for framework-specific trace parsers.

    Each subclass reads one framework's native trace format (LangGraph,
    AutoGen, CrewAI, ...) from a file and returns a `NormalizedTrace`.
    """

    #: Short identifier for the framework this parser handles, e.g. "langgraph".
    #: Used as `NormalizedTrace.source_framework` and for CLI framework selection.
    framework: str

    @abstractmethod
    def parse(self, path: Path) -> NormalizedTrace:
        """Parse a trace file at `path` into a `NormalizedTrace`."""
        raise NotImplementedError
