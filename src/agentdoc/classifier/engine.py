"""The MAST classification engine: NormalizedTrace -> ClassificationResult.

Uses an LLM-as-a-judge approach (following the paper's own methodology):
the full taxonomy plus the trace are sent to an LLM backend, which returns a
structured (schema-validated) list of flagged failure modes rather than free
text we'd have to parse ourselves.
"""

from __future__ import annotations

from typing import Any

from agentdoc.classifier.llm import (
    AnthropicBackend,
    GroqBackend,
    LLMBackend,
    LLMBackendError,
)
from agentdoc.classifier.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    chunk_trace,
)
from agentdoc.classifier.results import ClassificationResult, FlaggedFailure
from agentdoc.classifier.taxonomy import FailureMode, get_failure_mode
from agentdoc.parsers.schema import NormalizedTrace

TOOL_NAME = "report_mast_classification"

#: Backend names accepted by `build_backend()` and the `agentdoc diagnose
#: --backend` CLI flag, mapped to their `LLMBackend` implementation. "groq"
#: is the default: it's free-tier-friendly, which matters more day-to-day
#: than Anthropic's typically higher output quality.
BACKENDS: dict[str, type[LLMBackend]] = {
    "groq": GroqBackend,
    "anthropic": AnthropicBackend,
}
DEFAULT_BACKEND = "groq"


def build_backend(name: str = DEFAULT_BACKEND) -> LLMBackend:
    """Instantiate the named backend with its default model/config.

    Args:
        name: One of the keys in `BACKENDS` (currently "groq" or "anthropic").

    Raises:
        LLMBackendError: if `name` isn't a known backend, or if the backend's
            own construction fails (e.g. missing API key).
    """
    backend_cls = BACKENDS.get(name)
    if backend_cls is None:
        raise LLMBackendError(
            f"Unknown LLM backend {name!r}. Available: {', '.join(sorted(BACKENDS))}"
        )
    return backend_cls()

#: JSON Schema for the structured output we require from the LLM backend.
#: `failure_mode` is constrained to the 14 known MAST IDs so a malformed or
#: hallucinated mode name fails validation at the API level rather than
#: silently entering a report. Every object sets `additionalProperties:
#: False` and lists all its properties as `required` — Anthropic's tool-use
#: doesn't need this, but Groq's *strict* json_schema mode does, and
#: satisfying both keeps one schema usable by every backend.
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "flagged_failures": {
            "type": "array",
            "description": (
                "Every MAST failure mode identified in the trace. Empty if none."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "failure_mode": {
                        "type": "string",
                        "enum": [mode.value for mode in FailureMode],
                        "description": "The MAST failure mode ID, e.g. 'FM-2.4'.",
                    },
                    "turn_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "The [step N] turn indices that evidence this failure."
                        ),
                        "minItems": 1,
                    },
                    "justification": {
                        "type": "string",
                        "description": (
                            "Brief, specific explanation grounded in what "
                            "happened at the cited turns."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in this flag, from 0.0 to 1.0.",
                    },
                },
                "required": [
                    "failure_mode",
                    "turn_indices",
                    "justification",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["flagged_failures"],
    "additionalProperties": False,
}


class MastClassifier:
    """Runs the MAST LLM-as-a-judge classifier over a `NormalizedTrace`.

    Example:
        >>> classifier = MastClassifier()  # uses GROQ_API_KEY
        >>> result = classifier.classify(trace)
        >>> for failure in result:
        ...     print(failure.failure_mode, failure.justification)
    """

    def __init__(self, backend: LLMBackend | str | None = None) -> None:
        """Create a classifier.

        Args:
            backend: An `LLMBackend` instance, a backend name string (one of
                `BACKENDS`, e.g. "groq" or "anthropic"), or `None`. Defaults
                to `GroqBackend()` (name "groq"), which reads `GROQ_API_KEY`
                from the environment and raises `LLMBackendError` if it's
                missing.
        """
        if backend is None:
            self.backend = build_backend(DEFAULT_BACKEND)
        elif isinstance(backend, str):
            self.backend = build_backend(backend)
        else:
            self.backend = backend

    def classify(self, trace: NormalizedTrace) -> ClassificationResult:
        """Classify a trace, transparently chunking it if it's too long.

        Failures flagged across chunks are merged into a single result. If
        the same failure mode is flagged in multiple overlapping chunks
        (possible near chunk boundaries), all flags are kept as-is rather
        than deduplicated, since deduplication would require deciding which
        justification/turn set is authoritative — callers wanting a single
        verdict per mode can post-process `ClassificationResult`.
        """
        chunks = chunk_trace(trace)

        all_failures: list[FlaggedFailure] = []
        raw_responses: list[dict[str, Any]] = []

        for chunk in chunks:
            raw = self._classify_chunk(chunk)
            raw_responses.append(raw)
            all_failures.extend(self._parse_flagged_failures(raw))

        return ClassificationResult(
            flagged_failures=all_failures,
            model=getattr(self.backend, "model", None),
            raw_response={"chunks": raw_responses} if len(raw_responses) > 1 else raw_responses[0],
        )

    def _classify_chunk(self, chunk: NormalizedTrace) -> dict[str, Any]:
        user_prompt = build_user_prompt(chunk)
        return self.backend.classify(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_schema=OUTPUT_SCHEMA,
            tool_name=TOOL_NAME,
        )

    @staticmethod
    def _parse_flagged_failures(raw: dict[str, Any]) -> list[FlaggedFailure]:
        failures = []
        for item in raw.get("flagged_failures", []):
            try:
                failure_mode_id = item["failure_mode"]
            except KeyError as exc:
                # Should not happen under strict/tool-enforced schemas, but a
                # provider could still return a loosely-conforming response
                # missing a required field — surface it as a classifier error
                # rather than letting a raw KeyError escape to the caller.
                raise LLMBackendError(
                    f"LLM response item is missing 'failure_mode': {item!r}"
                ) from exc

            try:
                mode_info = get_failure_mode(failure_mode_id)
            except KeyError as exc:
                raise LLMBackendError(
                    f"LLM returned an unrecognized failure_mode: {item!r}"
                ) from exc

            try:
                failures.append(
                    FlaggedFailure(
                        failure_mode=mode_info.id,
                        category=mode_info.category,
                        turn_indices=list(item["turn_indices"]),
                        justification=item["justification"],
                        confidence=float(item["confidence"]),
                    )
                )
            except KeyError as exc:
                raise LLMBackendError(
                    f"LLM response item is missing an expected field {exc}: {item!r}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise LLMBackendError(
                    f"LLM response item has an unexpected shape: {item!r}"
                ) from exc
        return failures
