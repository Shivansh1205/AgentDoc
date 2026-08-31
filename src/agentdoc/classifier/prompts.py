"""Prompt construction for the MAST LLM-as-a-judge classifier.

Kept separate from `engine.py` so the prompt text and trace-rendering/chunking
logic can be unit-tested (and iterated on) without invoking any LLM backend.
"""

from __future__ import annotations

from agentdoc.classifier.taxonomy import (
    FAILURE_MODES,
    CATEGORY_NAMES,
    FailureCategory,
)
from agentdoc.parsers.schema import NormalizedTrace, Turn

# Rough chunking budget. This is a conservative character count (not a token
# count) intended to keep a single classification call comfortably within
# typical context windows once the taxonomy definitions and system prompt are
# included. A trace longer than this is split into overlapping chunks so each
# call still has some surrounding context.
DEFAULT_CHUNK_CHAR_BUDGET = 40_000
#: Number of turns of overlap between consecutive chunks, so a failure whose
#: evidence straddles a chunk boundary (e.g. step repetition) is still visible
#: with its preceding context in at least one chunk.
DEFAULT_CHUNK_OVERLAP_TURNS = 3


def build_taxonomy_reference() -> str:
    """Render the full MAST taxonomy (all 14 failure modes) as prompt text."""
    sections = []
    for category in FailureCategory:
        category_name = CATEGORY_NAMES[category]
        modes = [info for info in FAILURE_MODES.values() if info.category is category]
        modes.sort(key=lambda info: info.id.value)
        lines = [f"## {category_name}"]
        for info in modes:
            lines.append(f"- {info.id.value} - {info.name}: {info.definition}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def render_turn(turn: Turn) -> str:
    """Render one `Turn` as a single line/block for the trace transcript."""
    header = f"[step {turn.step}] "
    if turn.agent:
        header += f"{turn.agent} "
    header += f"({turn.role.value})"

    lines = [header]
    if turn.content:
        lines.append(turn.content)
    for call in turn.tool_calls:
        lines.append(f"  -> tool_call: {call.name}(args={call.args})")
        if call.result is not None:
            lines.append(f"     result: {call.result}")
        if call.error is not None:
            lines.append(f"     error: {call.error}")
    return "\n".join(lines)


def render_trace(trace: NormalizedTrace) -> str:
    """Render a full `NormalizedTrace` as a plain-text transcript."""
    return "\n\n".join(render_turn(turn) for turn in trace)


def chunk_trace(
    trace: NormalizedTrace,
    *,
    char_budget: int = DEFAULT_CHUNK_CHAR_BUDGET,
    overlap_turns: int = DEFAULT_CHUNK_OVERLAP_TURNS,
) -> list[NormalizedTrace]:
    """Split a trace into chunks small enough for one classification call.

    Each chunk is itself a `NormalizedTrace` (same `source_framework`/metadata,
    a contiguous slice of `turns`) so the rest of the pipeline can treat
    "one trace" and "one chunk of a trace" identically. Chunks overlap by
    `overlap_turns` turns so evidence near a boundary isn't classified with
    zero preceding context.

    If the whole trace already fits within `char_budget`, returns a single
    chunk containing the entire trace (i.e. chunking is a no-op for short
    traces, which is the common case).
    """
    if not trace.turns:
        return [trace]

    rendered_lengths = [len(render_turn(turn)) for turn in trace.turns]
    if sum(rendered_lengths) <= char_budget:
        return [trace]

    chunks: list[NormalizedTrace] = []
    start = 0
    n = len(trace.turns)
    while start < n:
        end = start
        length = 0
        while end < n and (length + rendered_lengths[end] <= char_budget or end == start):
            length += rendered_lengths[end]
            end += 1

        chunks.append(
            NormalizedTrace(
                turns=trace.turns[start:end],
                source_framework=trace.source_framework,
                metadata={**trace.metadata, "chunk_turn_range": (start, end - 1)},
            )
        )

        if end >= n:
            break
        start = max(end - overlap_turns, start + 1)

    return chunks


SYSTEM_PROMPT = """You are an expert judge diagnosing failures in multi-agent LLM systems.

You will be given an execution trace of a multi-agent system (a sequence of \
turns, each attributable to an agent, tool, or human) and the MAST \
(Multi-Agent System Failure Taxonomy) failure mode definitions.

Your task: identify every MAST failure mode that clearly occurred in the \
trace. For each one you flag, cite the specific turn(s) that evidence it and \
give a brief, concrete justification grounded in what actually happened in \
those turns — not a generic restatement of the failure mode's definition.

Guidelines:
- Only flag a failure mode when the trace provides clear, specific evidence \
for it. Do not flag modes speculatively or because a trace is short.
- A trace may exhibit zero, one, or several failure modes, in any category.
- Cite turn indices using the `[step N]` markers in the transcript.
- Confidence should reflect how unambiguous the evidence is, not how severe \
the failure is."""


def build_user_prompt(trace: NormalizedTrace) -> str:
    """Build the user-turn prompt: taxonomy reference + rendered trace."""
    taxonomy = build_taxonomy_reference()
    transcript = render_trace(trace)
    return (
        "# MAST Failure Mode Definitions\n\n"
        f"{taxonomy}\n\n"
        "# Execution Trace\n\n"
        f"{transcript}\n\n"
        "# Task\n\n"
        "Identify which MAST failure modes (if any) occurred in the trace "
        "above, following the guidelines in your instructions."
    )
