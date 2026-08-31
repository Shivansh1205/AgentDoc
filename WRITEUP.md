# I built a tool that tells you why your multi-agent LLM system failed

If you've built a multi-agent LLM system — a researcher agent handing off to
a writer agent, a supervisor coordinating specialists, anything with more
than one LLM call talking to another LLM call — you've probably hit this
moment: the system produces a wrong or incomplete answer, and you have no
idea *why*. Was it a bad tool call? Did one agent ignore what another agent
told it? Did something declare "done" before actually checking the work?

The honest answer, most of the time, is: you scroll through a wall of JSON
and agent chatter and guess.

## The problem is real, and it's been measured

This isn't just a vibe. A team at Berkeley (Cemri et al.) published
["Why Do Multi-Agent LLM Systems Fail?"](https://arxiv.org/abs/2503.13657),
annotating over 1,600 execution traces from 7 popular frameworks and finding
failure rates between 41% and 87%. More usefully, they didn't just measure
*that* these systems fail — they built a taxonomy of *how*: **MAST**
(Multi-Agent System Failure Taxonomy), 14 failure modes across 3 categories:

- **System Design Issues** — the system disobeyed its own task or role
  specification, repeated a step for no reason, lost conversation history,
  or never noticed it should have stopped.
- **Inter-Agent Misalignment** — agents talked past each other: one withheld
  information, ignored another's input, or its reasoning and its actions
  didn't match.
- **Task Verification** — nobody actually checked the output. The system
  terminated early, verified nothing, or verified the wrong thing.

Once I read that taxonomy, I couldn't unsee it in my own traces. The
question became: why am I doing this classification by eye, one trace at a
time, when it's a well-defined enough problem that an LLM-as-a-judge could
do it consistently?

## What AgentDoc does

AgentDoc is a CLI that takes a multi-agent execution trace and tells you
which of the 14 MAST failure modes actually occurred, with a justification
and the specific turns that caused it — instead of you re-reading the trace
line by line.

```bash
pip install agentdoc
export GROQ_API_KEY=...   # free at console.groq.com/keys
agentdoc diagnose my_trace.json
```

On a clean run, it says so plainly:

```
+------------------------------------------------------------------------------+
| MAST Diagnosis Summary  |  framework=langgraph  |  turns=5  |                |
| model=openai/gpt-oss-120b                                                    |
| This run shows no MAST failure modes: no issues were flagged.                |
+------------------------------------------------------------------------------+
```

On a trace where a writer agent contradicts what its researcher teammate
just confirmed, and a supervisor closes the task without checking anything:

```
System Design Issues: 2   Inter-Agent Misalignment: 1   Task Verification: 2

FM-2.5  Ignored other agent's input  |  confidence=0.93  |  turns=[5]
The writer ignored the researcher's input ($412M revenue) and supplied its
own contradictory number, demonstrating disregard for another agent's
contribution.
```

Under the hood: a parser converts a framework's native trace format
(LangGraph is supported now; AutoGen and CrewAI are next) into a normalized,
framework-agnostic representation, then an LLM judge — Groq's
`openai/gpt-oss-120b` by default, since it's free, or Claude via Anthropic
if you'd rather — classifies it against the full MAST taxonomy using
structured output, so you get back typed, schema-validated failure records
instead of free text to re-parse. You can also export the full report as
JSON (`--json report.json`) for scripting or CI gates.

## What's next

Right now AgentDoc's output is text — a terminal report or JSON. The
taxonomy naturally wants to be a graph, though: agents as nodes, handoffs
and tool calls as edges, failures highlighted right where they happened in
the flow. That's the next phase — an `--html` flag that renders the
diagnosed trace as an interactive visualization instead of a wall of panels.
The report data model is already structured for it; it's a rendering layer,
not a rewrite.

If you're building multi-agent systems and have a trace you're stuck on,
[give it a try](https://github.com/Shivansh1205/AgentDoc) — it's free to run
and takes about thirty seconds to get a diagnosis instead of a guess.
