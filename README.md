# AgentDoc

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Shivansh1205/AgentDoc/actions/workflows/ci.yml/badge.svg)](https://github.com/Shivansh1205/AgentDoc/actions/workflows/ci.yml)

**AgentDoc diagnoses *why* your multi-agent LLM system failed** — not just
that it did. Point it at an execution trace and it tells you which specific,
named failure pattern occurred: an agent ignored another agent's tool
result, the system declared "done" without checking the output, a step got
repeated for no reason, and so on.

Multi-agent systems (LangGraph, AutoGen, CrewAI, ...) fail *a lot*, and when
they do, the failure is usually buried in a wall of tool calls and agent
chatter. AgentDoc classifies that trace against **MAST** (Multi-Agent System
Failure Taxonomy) — a taxonomy of 14 concrete failure modes empirically
derived from Berkeley's research on why multi-agent LLM systems fail
([Cemri et al., "Why Do Multi-Agent LLM Systems Fail?", arXiv:2503.13657](https://arxiv.org/abs/2503.13657)) —
so instead of re-reading a trace line by line, you get a report that names
the problem.

<!--
DEMO GIF: record following demo/RECORDING.md, then replace this comment with:
![AgentDoc demo](demo/agentdoc-demo.gif)
-->

## Quickstart

```bash
pip install agentdoc
```

Get a free [Groq](https://console.groq.com/keys) API key (AgentDoc's default
backend — no cost, no card required) and set it:

```bash
export GROQ_API_KEY=gsk_...          # macOS/Linux
$env:GROQ_API_KEY = "gsk_..."        # Windows PowerShell
```

(Or put it in a `.env` file in your working directory — AgentDoc loads it
automatically.)

Then diagnose a trace:

```bash
agentdoc diagnose examples/langgraph_trace_example.json
```

## Example output

A clean run — the researcher looks something up once, the writer's answer
matches what was found, nobody's confused:

```
+------------------------------------------------------------------------------+
| MAST Diagnosis Summary  |  framework=langgraph  |  turns=5  |                |
| model=openai/gpt-oss-120b                                                    |
| This run shows no MAST failure modes: no issues were flagged.                |
+------------------------------------------------------------------------------+
```

A trace with real problems injected — a duplicated tool call, a writer that
contradicts what the researcher found, and a supervisor that closes the task
without checking anything:

```
+------------------------------------------------------------------------------+
| MAST Diagnosis Summary  |  framework=langgraph  |  turns=7  |                |
| model=openai/gpt-oss-120b                                                    |
| This run shows 5 failures.                                                   |
|                                                                              |
| System Design Issues: 2   Inter-Agent Misalignment: 1   Task Verification: 2 |
|                                                                              |
| Most frequent: Disobey task specification (FM-1.1) x1, Disobey role          |
| specification (FM-1.2) x1, Ignored other agent's input (FM-2.5) x1           |
+------------------------------------------------------------------------------+

System Design Issues
+------------------------------------------------------------------------------+
| FM-1.1  Disobey task specification  |  confidence=0.95  |  turns=[5]         |
| The writer answered with $380 million revenue, which contradicts the         |
| researcher's verified figure of $412 million, thus not adhering to the task  |
| of providing the correct revenue.                                            |
|                                                                              |
| [step 5] writer (agent)                                                      |
| Acme Corp's most recent quarterly revenue was $380 million, a slight decline |
| from the prior quarter. Let me know if you need any other financial details. |
+------------------------------------------------------------------------------+

Inter-Agent Misalignment
+------------------------------------------------------------------------------+
| FM-2.5  Ignored other agent's input  |  confidence=0.93  |  turns=[5]        |
| The writer ignored the researcher's input (the $412 M revenue) and supplied  |
| its own contradictory number, demonstrating disregard for another agent's    |
| contribution.                                                                |
|                                                                              |
| [step 5] writer (agent)                                                      |
| Acme Corp's most recent quarterly revenue was $380 million, a slight decline |
| from the prior quarter. Let me know if you need any other financial details. |
+------------------------------------------------------------------------------+

Task Verification
+------------------------------------------------------------------------------+
| FM-3.2  No or incomplete verification  |  confidence=0.96  |  turns=[6]      |
| The supervisor marked the task complete and closed the conversation without  |
| checking the writer's answer against the researcher's data, resulting in     |
| incomplete verification.                                                     |
|                                                                              |
| [step 6] supervisor (agent)                                                  |
| Writer has produced a response, so the task is marked complete. Closing the  |
| conversation without further review.                                         |
+------------------------------------------------------------------------------+
```

*(Real captured output from `examples/langgraph_trace_flawed_example.json`,
trimmed to 3 of the 5 flagged failures for length. Classification is
LLM-based, so exact wording, confidence scores, and which specific failure
mode gets flagged can vary slightly between runs — see
[`examples/README.md`](examples/README.md) for what's deliberately injected
into that fixture.)*

## The MAST taxonomy

MAST groups 14 failure modes into 3 categories, based on *where* in a
multi-agent system's design the root cause sits:

| Category | What it covers | Failure modes |
|---|---|---|
| **System Design Issues** | Problems baked into how the system was built — prompts, roles, state handling | Disobey task/role specification, step repetition, loss of conversation history, unaware of termination conditions |
| **Inter-Agent Misalignment** | Breakdowns in how agents coordinate with each other | Conversation reset, fail to ask for clarification, task derailment, information withholding, ignored input, reasoning-action mismatch |
| **Task Verification** | Failures to properly check that the output is actually correct/complete | Premature termination, no or incomplete verification, incorrect verification |

Full names and definitions (verbatim from the paper's Appendix A) live in
[`src/agentdoc/classifier/taxonomy.py`](src/agentdoc/classifier/taxonomy.py).

## Supported frameworks

- **LangGraph** — supported now, via its stream-capture trace format.
- **AutoGen, CrewAI** — planned. Each framework gets its own parser under
  [`src/agentdoc/parsers/`](src/agentdoc/parsers/) that outputs the same
  framework-agnostic normalized trace, so the classifier and report layers
  never need to know which framework produced a trace.

## LLM backend & API key

Classification calls an LLM to judge a trace against the MAST taxonomy. Two
backends are supported:

- **Groq (default, free)** — set `GROQ_API_KEY`. Get a free key at
  [console.groq.com/keys](https://console.groq.com/keys). Uses
  `openai/gpt-oss-120b`, available on Groq's free tier with strict JSON
  Schema structured output.
- **Anthropic (optional, paid)** — set `ANTHROPIC_API_KEY` and pass
  `--backend anthropic`. Uses Claude with forced tool-use for structured
  output.

Either way, trace content is sent only to whichever backend you choose —
pick the one whose data handling you're comfortable with for your traces.

## Usage

```bash
agentdoc --version

# Parse a trace into AgentDoc's normalized format and inspect it
agentdoc parse examples/langgraph_trace_example.json

# Diagnose against the MAST taxonomy (Groq by default)
agentdoc diagnose examples/langgraph_trace_example.json

# Use Anthropic instead
agentdoc diagnose examples/langgraph_trace_example.json --backend anthropic

# Also export the full structured report as JSON
agentdoc diagnose examples/langgraph_trace_example.json --json report.json

# JSON only, no terminal report (e.g. for scripting/CI)
agentdoc diagnose examples/langgraph_trace_example.json --json report.json --json-only
```

Run `agentdoc --help` or `agentdoc diagnose --help` for the full flag
reference.

## Installing from source / development

```bash
git clone <this-repo>
cd AgentDoc
uv sync --group dev   # installs runtime + dev (pytest) dependencies
uv run agentdoc --version
uv run pytest
```

This project uses [`uv`](https://docs.astral.sh/uv/) for local dependency
management, but the published package is a standard `pip`-installable
project — `pip install agentdoc`, `pip install .`, and `pipx install .` all
work without `uv` present.

## Project layout

```
src/agentdoc/
  parsers/     # Framework-specific trace parsers -> normalized trace format
  classifier/  # MAST taxonomy + LLM-as-a-judge engine (pluggable Groq/Anthropic backends)
  report/      # ReportSummary generation, terminal rendering, JSON export
tests/         # Test suite (mocked LLM calls only - no real API usage in tests)
examples/      # Sample trace files, including a deliberately flawed regression fixture
```
