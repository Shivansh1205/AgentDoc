"""Tests for the MAST classification engine, using a mocked LLM backend.

No test in this module hits a real LLM API. `FakeLLMBackend` implements the
`LLMBackend` protocol directly and returns canned structured responses, so
these tests exercise the engine's prompt-building, response-parsing, and
chunking logic in isolation.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentdoc.classifier.engine import BACKENDS, DEFAULT_BACKEND, MastClassifier, build_backend
from agentdoc.classifier.llm import AnthropicBackend, GroqBackend, LLMBackendError
from agentdoc.classifier.results import ClassificationResult
from agentdoc.classifier.taxonomy import FailureCategory, FailureMode
from agentdoc.parsers.schema import NormalizedTrace, Role, ToolCall, Turn


class FakeLLMBackend:
    """A stand-in `LLMBackend` that returns a pre-programmed structured response.

    Records the prompts it was called with so tests can assert on prompt
    construction without depending on exact prompt wording.
    """

    model = "fake-model-1"

    def __init__(self, response: dict[str, Any] | list[dict[str, Any]]) -> None:
        # A single dict is returned for every call; a list is returned once
        # per call, in order (used to test multi-chunk merging).
        self._responses = response if isinstance(response, list) else None
        self._single_response = response if not isinstance(response, list) else None
        self.calls: list[dict[str, Any]] = []

    def classify(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "output_schema": output_schema,
                "tool_name": tool_name,
            }
        )
        if self._responses is not None:
            return self._responses[len(self.calls) - 1]
        return self._single_response


@pytest.fixture
def flawed_trace() -> NormalizedTrace:
    """A deliberately flawed trace: an agent ignores a tool result and
    contradicts itself, then a second agent proceeds without seeking
    clarification about the contradiction.

    This fixture exercises the prompt design conceptually (it's the kind of
    trace a real judge should flag FM-2.5 / FM-2.6 / FM-2.2 on) but the LLM
    call itself is always mocked — see `FakeLLMBackend` above.
    """
    return NormalizedTrace(
        source_framework="langgraph",
        turns=[
            Turn(
                step=0,
                role=Role.HUMAN,
                content="What is the current version of the widget spec?",
            ),
            Turn(
                step=1,
                role=Role.AGENT,
                agent="researcher",
                content="Let me check.",
                tool_calls=[
                    ToolCall(
                        name="lookup_spec",
                        call_id="call_1",
                        args={"doc": "widget-spec"},
                        result="Version 3.2, published last week.",
                    )
                ],
            ),
            Turn(
                step=2,
                role=Role.AGENT,
                agent="researcher",
                content=(
                    "The widget spec is version 1.0, which has been stable "
                    "for years."
                ),
            ),
            Turn(
                step=3,
                role=Role.AGENT,
                agent="writer",
                content="Great, drafting the doc based on version 1.0 now.",
            ),
        ],
    )


def _sample_raw_response() -> dict[str, Any]:
    return {
        "flagged_failures": [
            {
                "failure_mode": "FM-2.6",
                "turn_indices": [1, 2],
                "justification": (
                    "The researcher's tool call returned 'version 3.2' but "
                    "the agent's next message asserts 'version 1.0', "
                    "contradicting its own retrieved evidence."
                ),
                "confidence": 0.9,
            },
            {
                "failure_mode": "FM-2.2",
                "turn_indices": [3],
                "justification": (
                    "The writer proceeds on the researcher's claim without "
                    "questioning the discrepancy with commonly known info."
                ),
                "confidence": 0.6,
            },
        ]
    }


def test_classify_parses_response_into_classification_result(
    flawed_trace: NormalizedTrace,
) -> None:
    backend = FakeLLMBackend(_sample_raw_response())
    classifier = MastClassifier(backend=backend)

    result = classifier.classify(flawed_trace)

    assert isinstance(result, ClassificationResult)
    assert len(result) == 2
    assert result.model == "fake-model-1"


def test_flagged_failure_fields_are_correctly_mapped(
    flawed_trace: NormalizedTrace,
) -> None:
    backend = FakeLLMBackend(_sample_raw_response())
    classifier = MastClassifier(backend=backend)

    result = classifier.classify(flawed_trace)
    first = result.flagged_failures[0]

    assert first.failure_mode is FailureMode.REASONING_ACTION_MISMATCH
    assert first.category is FailureCategory.INTER_AGENT_MISALIGNMENT
    assert first.turn_indices == [1, 2]
    assert "contradicting" in first.justification
    assert first.confidence == pytest.approx(0.9)


def test_category_is_derived_not_trusted_from_llm(flawed_trace: NormalizedTrace) -> None:
    # The engine must derive `category` from the taxonomy, not accept it
    # verbatim from the LLM (the schema doesn't even ask the LLM for it).
    backend = FakeLLMBackend(_sample_raw_response())
    classifier = MastClassifier(backend=backend)

    result = classifier.classify(flawed_trace)
    clarification_failure = result.flagged_failures[1]

    assert clarification_failure.failure_mode is FailureMode.FAIL_TO_ASK_FOR_CLARIFICATION
    assert clarification_failure.category is FailureCategory.INTER_AGENT_MISALIGNMENT


def test_empty_flagged_failures_is_valid(flawed_trace: NormalizedTrace) -> None:
    backend = FakeLLMBackend({"flagged_failures": []})
    classifier = MastClassifier(backend=backend)

    result = classifier.classify(flawed_trace)

    assert len(result) == 0
    assert list(result) == []


def test_unrecognized_failure_mode_raises(flawed_trace: NormalizedTrace) -> None:
    backend = FakeLLMBackend(
        {
            "flagged_failures": [
                {
                    "failure_mode": "FM-9.9",  # not a real MAST mode
                    "turn_indices": [0],
                    "justification": "bogus",
                    "confidence": 0.5,
                }
            ]
        }
    )
    classifier = MastClassifier(backend=backend)

    with pytest.raises(LLMBackendError, match="unrecognized failure_mode"):
        classifier.classify(flawed_trace)


def test_missing_failure_mode_field_raises_llm_backend_error(
    flawed_trace: NormalizedTrace,
) -> None:
    # A loosely-conforming response missing the failure_mode key entirely
    # should surface as LLMBackendError, not a raw KeyError.
    backend = FakeLLMBackend(
        {
            "flagged_failures": [
                {"turn_indices": [0], "justification": "bogus", "confidence": 0.5}
            ]
        }
    )
    classifier = MastClassifier(backend=backend)

    with pytest.raises(LLMBackendError, match="missing 'failure_mode'"):
        classifier.classify(flawed_trace)


def test_missing_other_field_raises_llm_backend_error(
    flawed_trace: NormalizedTrace,
) -> None:
    backend = FakeLLMBackend(
        {
            "flagged_failures": [
                {"failure_mode": "FM-2.6", "turn_indices": [0], "confidence": 0.5}
                # missing "justification"
            ]
        }
    )
    classifier = MastClassifier(backend=backend)

    with pytest.raises(LLMBackendError, match="missing an expected field"):
        classifier.classify(flawed_trace)


def test_malformed_confidence_raises_llm_backend_error(
    flawed_trace: NormalizedTrace,
) -> None:
    backend = FakeLLMBackend(
        {
            "flagged_failures": [
                {
                    "failure_mode": "FM-2.6",
                    "turn_indices": [0],
                    "justification": "bogus",
                    "confidence": "not-a-number",
                }
            ]
        }
    )
    classifier = MastClassifier(backend=backend)

    with pytest.raises(LLMBackendError, match="unexpected shape"):
        classifier.classify(flawed_trace)


def test_prompt_includes_taxonomy_and_trace_content(flawed_trace: NormalizedTrace) -> None:
    backend = FakeLLMBackend({"flagged_failures": []})
    classifier = MastClassifier(backend=backend)

    classifier.classify(flawed_trace)

    assert len(backend.calls) == 1
    call = backend.calls[0]
    # Taxonomy reference must be present so the judge has definitions to work from.
    assert "FM-2.6" in call["user_prompt"]
    assert "Reasoning-action mismatch" in call["user_prompt"]
    # The actual trace content must be present, not just the taxonomy.
    assert "widget spec" in call["user_prompt"]
    assert "Version 3.2" in call["user_prompt"]
    # Structured output must be forced via schema + a named tool.
    assert call["tool_name"] == "report_mast_classification"
    assert call["output_schema"]["type"] == "object"


def test_by_category_filters_correctly(flawed_trace: NormalizedTrace) -> None:
    backend = FakeLLMBackend(_sample_raw_response())
    classifier = MastClassifier(backend=backend)

    result = classifier.classify(flawed_trace)
    misalignment_failures = result.by_category(FailureCategory.INTER_AGENT_MISALIGNMENT)
    verification_failures = result.by_category(FailureCategory.TASK_VERIFICATION)

    assert len(misalignment_failures) == 2
    assert len(verification_failures) == 0


def test_multi_chunk_traces_merge_failures_from_all_chunks(
    flawed_trace: NormalizedTrace,
) -> None:
    # Force chunking regardless of trace length by monkeypatching chunk_trace
    # indirectly: simulate two chunks by giving the backend two responses and
    # verifying the engine calls it twice when chunk_trace splits the trace.
    import agentdoc.classifier.engine as engine_module

    chunk_a = NormalizedTrace(turns=flawed_trace.turns[:2], source_framework="langgraph")
    chunk_b = NormalizedTrace(turns=flawed_trace.turns[2:], source_framework="langgraph")

    original_chunk_trace = engine_module.chunk_trace
    engine_module.chunk_trace = lambda trace, **kwargs: [chunk_a, chunk_b]
    try:
        backend = FakeLLMBackend(
            [
                {
                    "flagged_failures": [
                        {
                            "failure_mode": "FM-2.6",
                            "turn_indices": [1, 2],
                            "justification": "chunk a finding",
                            "confidence": 0.9,
                        }
                    ]
                },
                {
                    "flagged_failures": [
                        {
                            "failure_mode": "FM-2.2",
                            "turn_indices": [3],
                            "justification": "chunk b finding",
                            "confidence": 0.6,
                        }
                    ]
                },
            ]
        )
        classifier = MastClassifier(backend=backend)
        result = classifier.classify(flawed_trace)

        assert len(backend.calls) == 2
        assert len(result) == 2
        modes = {f.failure_mode for f in result}
        assert modes == {
            FailureMode.REASONING_ACTION_MISMATCH,
            FailureMode.FAIL_TO_ASK_FOR_CLARIFICATION,
        }
    finally:
        engine_module.chunk_trace = original_chunk_trace


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMBackendError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend()


# ---------------------------------------------------------------------------
# Backend selection (build_backend / MastClassifier(backend=<name>))
# ---------------------------------------------------------------------------


def test_default_backend_is_groq() -> None:
    assert DEFAULT_BACKEND == "groq"


def test_backends_registry_has_groq_and_anthropic() -> None:
    assert BACKENDS == {"groq": GroqBackend, "anthropic": AnthropicBackend}


def test_build_backend_defaults_to_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = build_backend()
    assert isinstance(backend, GroqBackend)


def test_build_backend_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    backend = build_backend("anthropic")
    assert isinstance(backend, AnthropicBackend)


def test_build_backend_rejects_unknown_name() -> None:
    with pytest.raises(LLMBackendError, match="Unknown LLM backend 'openai'"):
        build_backend("openai")


def test_mast_classifier_accepts_backend_instance(flawed_trace: NormalizedTrace) -> None:
    backend = FakeLLMBackend({"flagged_failures": []})
    classifier = MastClassifier(backend=backend)
    assert classifier.backend is backend


def test_mast_classifier_accepts_backend_name_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    classifier = MastClassifier(backend="groq")
    assert isinstance(classifier.backend, GroqBackend)


def test_mast_classifier_defaults_to_groq_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    classifier = MastClassifier()
    assert isinstance(classifier.backend, GroqBackend)
