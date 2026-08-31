"""Structured output schema for the MAST classifier.

Kept separate from `taxonomy.py` (the static reference data) and `engine.py`
(the orchestration logic) so the result shape can be imported by report
generation without pulling in the LLM-calling machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentdoc.classifier.taxonomy import FailureCategory, FailureMode


@dataclass
class FlaggedFailure:
    """One MAST failure mode the classifier believes occurred in a trace.

    Attributes:
        failure_mode: Which of the 14 MAST failure modes was flagged.
        category: The failure mode's top-level MAST category (denormalized
            from `failure_mode` for convenience — always consistent with it).
        turn_indices: The `Turn.step` index(es) in the source `NormalizedTrace`
            where this failure is evidenced. Usually one turn, but some modes
            (e.g. step repetition, reasoning-action mismatch) span turns.
        justification: Brief, specific explanation of why this failure mode
            applies at these turns, as produced by the judge.
        confidence: The judge's self-reported confidence in this flag, in
            [0.0, 1.0]. Not calibrated against ground truth — treat as a
            relative ranking signal, not a probability.
    """

    failure_mode: FailureMode
    category: FailureCategory
    turn_indices: list[int]
    justification: str
    confidence: float


@dataclass
class ClassificationResult:
    """The full output of running the MAST classifier on one trace.

    Attributes:
        flagged_failures: All failure modes the judge identified, in the
            order returned by the model (not otherwise sorted).
        model: Identifier of the LLM backend/model used to produce this
            result (e.g. "claude-sonnet-5"), for provenance.
        raw_response: The unparsed structured response from the LLM backend,
            kept for debugging prompt/schema issues. Not intended for display.
    """

    flagged_failures: list[FlaggedFailure] = field(default_factory=list)
    model: str | None = None
    raw_response: dict | None = None

    def __len__(self) -> int:
        return len(self.flagged_failures)

    def __iter__(self):
        return iter(self.flagged_failures)

    def by_category(self, category: FailureCategory) -> list[FlaggedFailure]:
        """All flagged failures belonging to `category`."""
        return [f for f in self.flagged_failures if f.category is category]
