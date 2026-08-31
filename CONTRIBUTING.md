# Contributing to AgentDoc

Thanks for considering a contribution. This project is early-stage, so
expect some rough edges — issues and PRs are both welcome.

## Development setup

AgentDoc uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/Shivansh1205/AgentDoc.git
cd AgentDoc
uv sync --group dev
```

This creates a `.venv/` with both runtime and dev (pytest) dependencies.

To run a command inside the environment without activating it:

```bash
uv run agentdoc --version
```

## Running tests

```bash
uv run pytest
```

Tests never call a real LLM API — every classifier/backend test mocks the
LLM call (see `tests/test_llm_backends.py` and `tests/test_classifier_engine.py`
for the pattern). If you add code that calls `GroqBackend`/`AnthropicBackend`
or anything downstream of `LLMBackend`, mock it the same way rather than
hitting the network in a test.

Run a single file or test while iterating:

```bash
uv run pytest tests/test_langgraph_parser.py -v
uv run pytest tests/test_cli.py::test_version_flag -v
```

## Project structure

```
src/agentdoc/
  parsers/     # Framework-specific trace parsers -> normalized trace format
  classifier/  # MAST taxonomy + LLM-as-a-judge engine (pluggable backends)
  report/      # ReportSummary generation, terminal rendering, JSON export
  cli.py       # Typer CLI wiring
tests/         # One test file per module, generally mirroring src/ layout
examples/      # Sample trace files, incl. a deliberately flawed regression fixture
```

## Adding a new framework parser

AgentDoc's classifier and report code only ever see the framework-agnostic
`NormalizedTrace` format (`agentdoc.parsers.schema`) — they have no idea
whether a trace came from LangGraph, AutoGen, or anything else. Adding
support for a new framework means writing one parser that converts its
native trace format into that shape. LangGraph
(`src/agentdoc/parsers/langgraph_parser.py`) is the reference implementation
to copy from.

Steps:

1. **Implement the `TraceParser` ABC** (`agentdoc.parsers.base`) in a new
   file, `src/agentdoc/parsers/<framework>_parser.py`:

   ```python
   from pathlib import Path
   from agentdoc.parsers.base import TraceParser
   from agentdoc.parsers.schema import NormalizedTrace

   class MyFrameworkParser(TraceParser):
       framework = "myframework"  # used as --framework value and source_framework

       def parse(self, path: Path) -> NormalizedTrace:
           # Read the framework's native trace format from `path` and
           # return a NormalizedTrace built from Turn/ToolCall/Role objects.
           ...
   ```

   Study `agentdoc/parsers/schema.py` for the `Turn`, `ToolCall`, and `Role`
   dataclasses you'll be constructing, and `langgraph_parser.py`'s module
   docstring for how it documents the exact trace shape it expects — do the
   same for your framework so the format is unambiguous for future readers.

2. **Register it** in `src/agentdoc/parsers/__init__.py`'s `PARSERS` dict:

   ```python
   from agentdoc.parsers.myframework_parser import MyFrameworkParser

   PARSERS: dict[str, type[TraceParser]] = {
       LangGraphParser.framework: LangGraphParser,
       MyFrameworkParser.framework: MyFrameworkParser,
   }
   ```

   This is the only wiring needed — `agentdoc parse --framework myframework`
   and `agentdoc diagnose --framework myframework` pick it up automatically.

3. **Raise clear errors on malformed input.** Follow `LangGraphParseError`'s
   pattern: a dedicated exception subclassing `ValueError` (or similar),
   with messages specific enough that a user can fix their trace file
   without reading your parser's source.

4. **Add a fixture and tests.** Add `examples/<framework>_trace_example.json`
   (or whatever your framework's native format is) and a
   `tests/test_<framework>_parser.py` mirroring
   `tests/test_langgraph_parser.py`'s coverage: happy path, malformed input,
   edge cases specific to that framework's trace format.

5. **Update the README's "Supported frameworks" section** once it's merged.

## Code style

- Match the surrounding code's conventions (naming, docstring density,
  comment style) rather than introducing a new style in one file.
- Prefer dataclasses over dicts for structured data crossing module
  boundaries (see `agentdoc.parsers.schema`, `agentdoc.classifier.results`,
  `agentdoc.report.summary`).
- Keep data/schema modules free of rendering logic — `ReportSummary` and
  friends are consumed by both the terminal renderer and the JSON exporter
  (and future output formats), so they should stay plain, serializable data
  rather than growing format-specific methods.

## Reporting bugs

Please use the bug report issue template — it asks for the framework and
backend you were using and a trace snippet, which is almost always what's
needed to reproduce a classifier or parser issue.
