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
