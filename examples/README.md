# Example traces

Both files are LangGraph "simplified stream capture" traces (see
`src/agentdoc/parsers/langgraph_parser.py` for the exact format) using the
same researcher -> writer scenario, so they're easy to diff against each
other.

## `langgraph_trace_example.json` — clean

A well-behaved run: researcher looks up a fact once, hands off cleanly, and
the writer's answer matches what the researcher found. `agentdoc diagnose`
should report zero flagged failures on this file. Useful as a smoke test /
sanity check that the classifier doesn't hallucinate problems on good runs.

## `langgraph_trace_flawed_example.json` — deliberately flawed

A regression fixture for the MAST classifier. It has the same
researcher -> writer -> supervisor shape as the clean example, but three
failures are deliberately injected:

1. **Step repetition (FM-1.3, System Design Issues)** — the researcher runs
   the *identical* `web_search` query twice in a row (steps 2-3) and gets
   the same result both times, with no new information gained. This models
   an agent redoing completed work rather than genuinely re-verifying with a
   different source or query.

2. **Ignored other agent's input (FM-2.5, Inter-Agent Misalignment)** — the
   writer's final answer (step 5, "$380 million, a slight decline") flatly
   contradicts the figure the researcher explicitly confirmed twice
   ($412 million, up 8%). The writer does not reference or reconcile the
   researcher's finding at all.

3. **Premature termination / no-or-incomplete verification (FM-3.1 and
   FM-3.2, Task Verification)** — the supervisor (step 6) marks the task
   complete and closes the conversation solely because *a* response was
   produced, without checking that the writer's answer actually matches the
   researcher's data or answers the original question.

As of the classifier prompt/taxonomy at the time this fixture was written,
`agentdoc diagnose examples/langgraph_trace_flawed_example.json` reliably
flags all three intended categories (System Design Issues, Inter-Agent
Misalignment, Task Verification), typically landing on FM-1.3, FM-2.5, and
one or both of FM-3.1/FM-3.2 — plus occasionally FM-1.1/FM-1.2 as additional,
reasonable catches on the same writer contradiction. Exact wording and
confidence scores vary run-to-run since classification is LLM-based, but a
healthy run of AgentDoc should **never report zero failures** on this file.
If it ever does, that's a signal the classifier or prompt has regressed.

### Using it as a regression check

```bash
uv run agentdoc diagnose examples/langgraph_trace_flawed_example.json
```

Expect: failures reported in all three MAST categories. Expect zero on
`langgraph_trace_example.json` as the negative control.

## `langgraph_swarm_example.json` — real external trace (langgraph-swarm-py)

Unlike the two fixtures above, **this is a real, unmodified capture from an
actual external multi-agent library**, not something we wrote by hand. It
was produced by running the `customer_support` example from
[`langgraph-swarm-py`](https://github.com/langchain-ai/langgraph-swarm-py)
(swapped from OpenAI to Groq's `openai/gpt-oss-120b` — the library itself is
model-agnostic) and capturing its actual `.stream(stream_mode="updates")`
output verbatim, with only cosmetic re-serialization (a `__type__` field
added for readability; harmless, the parser ignores unknown keys).

The trace: a user asks to book a flight and a hotel in one request. A
`flight_assistant` agent searches and books the flight, then **explicitly
hands off** to a `hotel_assistant` agent via `langgraph-swarm`'s
`transfer_to_hotel_assistant` tool call, which searches and books the hotel
and gives the final answer. 7 turns, 2 agents, 1 real structural handoff.

This fixture exists to test something the two synthetic fixtures above
can't: **compatibility with a trace shape AgentDoc didn't invent.**
`langgraph-swarm-py`'s raw `.stream()` output uses a different envelope
than our original "simplified stream capture" format (`{"<node_name>":
{...state}}` instead of `{"step", "node", "state", "timestamp"}`), and
represents handoffs as a specific tool-call convention
(`transfer_to_<agent_name>`) rather than anything AgentDoc's schema
originally had a concept for. Getting this file to parse correctly is what
drove two real fixes:

1. `LangGraphParser` now detects and accepts both envelope shapes (it used
   to silently produce zero turns on this file's shape rather than error).
2. `Turn.handoff_to` is a first-class, nullable field populated when a
   message's tool calls match the `transfer_to_<agent>` convention, so
   `agentdoc.report.html.build_graph()` can draw the handoff as a
   structurally-known edge (rendered with a distinct arrowhead/weight in
   the `--html` graph) instead of merely inferring it from consecutive
   turns having different agents.

### Using it as a regression check

```bash
uv run agentdoc parse examples/langgraph_swarm_example.json
```

Expect: 7 turns, `flight_assistant` (steps 1-3) then `hotel_assistant`
(steps 4-6), with `search_flights`/`book_flight`/`transfer_to_hotel_assistant`
folded correctly into `flight_assistant`'s turns and
`search_hotels`/`book_hotel` into `hotel_assistant`'s. If this ever parses
to 0 turns or raises, envelope detection has regressed. See
`tests/test_langgraph_parser.py`'s `TestLangGraphSwarmExample`-style tests
for the exact assertions (turn count, handoff_to population, tool-call
folding) that pin this down.
