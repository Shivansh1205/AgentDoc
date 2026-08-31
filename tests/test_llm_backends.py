"""Tests for the concrete LLMBackend implementations (Anthropic, Groq).

No test in this module makes a real network call. Each backend's underlying
SDK client is replaced with a fake/mock object before `classify()` is
invoked, so we exercise the backend's request construction and response
parsing in isolation — the same principle as `test_classifier_engine.py`'s
`FakeLLMBackend`, just one layer deeper (mocking the vendor SDK rather than
`LLMBackend` itself).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agentdoc.classifier.llm import AnthropicBackend, GroqBackend, LLMBackendError

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "flagged_failures": {
            "type": "array",
            "items": {"type": "object"},
        }
    },
    "required": ["flagged_failures"],
    "additionalProperties": False,
}

SAMPLE_RESULT = {
    "flagged_failures": [
        {
            "failure_mode": "FM-2.6",
            "turn_indices": [1, 2],
            "justification": "contradicted its own tool result",
            "confidence": 0.9,
        }
    ]
}


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://example.invalid/v1/chat")


# ---------------------------------------------------------------------------
# AnthropicBackend
# ---------------------------------------------------------------------------


def test_anthropic_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMBackendError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend()


def test_anthropic_backend_parses_tool_use_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    backend = AnthropicBackend(model="claude-sonnet-5")

    tool_use_block = SimpleNamespace(
        type="tool_use", name="report_mast_classification", input=SAMPLE_RESULT
    )
    fake_response = SimpleNamespace(content=[tool_use_block])
    backend._client = MagicMock()
    backend._client.messages.create.return_value = fake_response

    result = backend.classify(
        system_prompt="system",
        user_prompt="user",
        output_schema=SAMPLE_SCHEMA,
        tool_name="report_mast_classification",
    )

    assert result == SAMPLE_RESULT
    call_kwargs = backend._client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert call_kwargs["tool_choice"] == {
        "type": "tool",
        "name": "report_mast_classification",
    }
    assert call_kwargs["tools"][0]["input_schema"] == SAMPLE_SCHEMA


def test_anthropic_backend_raises_if_no_matching_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    backend = AnthropicBackend()

    text_block = SimpleNamespace(type="text", text="I refuse to use the tool.")
    backend._client = MagicMock()
    backend._client.messages.create.return_value = SimpleNamespace(content=[text_block])

    with pytest.raises(LLMBackendError, match="did not contain the expected tool call"):
        backend.classify(
            system_prompt="s",
            user_prompt="u",
            output_schema=SAMPLE_SCHEMA,
            tool_name="report_mast_classification",
        )


def test_anthropic_backend_wraps_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    backend = AnthropicBackend()

    backend._client = MagicMock()
    backend._client.messages.create.side_effect = anthropic.APIError(
        "boom", request=_fake_request(), body=None
    )

    with pytest.raises(LLMBackendError, match="Anthropic API request failed"):
        backend.classify(
            system_prompt="s",
            user_prompt="u",
            output_schema=SAMPLE_SCHEMA,
            tool_name="report_mast_classification",
        )


# ---------------------------------------------------------------------------
# GroqBackend
# ---------------------------------------------------------------------------


def test_groq_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMBackendError, match="GROQ_API_KEY"):
        GroqBackend()


def test_groq_backend_defaults_to_gpt_oss_120b(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = GroqBackend()
    assert backend.model == "openai/gpt-oss-120b"


def test_groq_backend_parses_json_schema_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = GroqBackend()

    fake_message = SimpleNamespace(content=json.dumps(SAMPLE_RESULT))
    fake_choice = SimpleNamespace(message=fake_message)
    fake_response = SimpleNamespace(choices=[fake_choice])
    backend._client = MagicMock()
    backend._client.chat.completions.create.return_value = fake_response

    result = backend.classify(
        system_prompt="system",
        user_prompt="user",
        output_schema=SAMPLE_SCHEMA,
        tool_name="report_mast_classification",
    )

    assert result == SAMPLE_RESULT
    call_kwargs = backend._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-oss-120b"
    response_format = call_kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == SAMPLE_SCHEMA
    # No tool-calling params: Groq doesn't support combining response_format
    # with tools/tool_choice, so this backend must not pass either.
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


def test_groq_backend_raises_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = GroqBackend()

    fake_message = SimpleNamespace(content=None)
    fake_choice = SimpleNamespace(message=fake_message)
    backend._client = MagicMock()
    backend._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[fake_choice]
    )

    with pytest.raises(LLMBackendError, match="did not contain any message content"):
        backend.classify(
            system_prompt="s",
            user_prompt="u",
            output_schema=SAMPLE_SCHEMA,
            tool_name="report_mast_classification",
        )


def test_groq_backend_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = GroqBackend()

    fake_message = SimpleNamespace(content="not valid json {{{")
    fake_choice = SimpleNamespace(message=fake_message)
    backend._client = MagicMock()
    backend._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[fake_choice]
    )

    with pytest.raises(LLMBackendError, match="was not valid JSON"):
        backend.classify(
            system_prompt="s",
            user_prompt="u",
            output_schema=SAMPLE_SCHEMA,
            tool_name="report_mast_classification",
        )


def test_groq_backend_wraps_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import groq

    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = GroqBackend()

    backend._client = MagicMock()
    backend._client.chat.completions.create.side_effect = groq.APIError(
        "boom", request=_fake_request(), body=None
    )

    with pytest.raises(LLMBackendError, match="Groq API request failed"):
        backend.classify(
            system_prompt="s",
            user_prompt="u",
            output_schema=SAMPLE_SCHEMA,
            tool_name="report_mast_classification",
        )


def _json_validate_failed_error() -> "groq.BadRequestError":
    import groq

    request = _fake_request()
    response = httpx.Response(400, request=request)
    body = {
        "error": {
            "message": "Failed to generate JSON. Please adjust your prompt.",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
        }
    }
    return groq.BadRequestError(
        "Failed to generate JSON.", response=response, body=body
    )


def test_groq_backend_retries_transient_json_validate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Groq occasionally emits a malformed value (e.g. "0. nine" for a number)
    # that fails its own strict schema validation server-side. This is a
    # transient sampling issue worth retrying, unlike a genuine API error.
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = GroqBackend()

    fake_message = SimpleNamespace(content=json.dumps(SAMPLE_RESULT))
    fake_choice = SimpleNamespace(message=fake_message)
    fake_response = SimpleNamespace(choices=[fake_choice])

    backend._client = MagicMock()
    backend._client.chat.completions.create.side_effect = [
        _json_validate_failed_error(),
        fake_response,
    ]

    result = backend.classify(
        system_prompt="s",
        user_prompt="u",
        output_schema=SAMPLE_SCHEMA,
        tool_name="report_mast_classification",
    )

    assert result == SAMPLE_RESULT
    assert backend._client.chat.completions.create.call_count == 2


def test_groq_backend_gives_up_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = GroqBackend()

    backend._client = MagicMock()
    backend._client.chat.completions.create.side_effect = _json_validate_failed_error()

    with pytest.raises(LLMBackendError, match="Groq API request failed"):
        backend.classify(
            system_prompt="s",
            user_prompt="u",
            output_schema=SAMPLE_SCHEMA,
            tool_name="report_mast_classification",
        )

    assert (
        backend._client.chat.completions.create.call_count
        == GroqBackend._MAX_GENERATION_ATTEMPTS
    )


def test_groq_backend_does_not_retry_non_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import groq

    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    backend = GroqBackend()

    backend._client = MagicMock()
    backend._client.chat.completions.create.side_effect = groq.AuthenticationError(
        "invalid api key",
        response=httpx.Response(401, request=_fake_request()),
        body={"error": {"message": "invalid api key", "code": "invalid_api_key"}},
    )

    with pytest.raises(LLMBackendError, match="Groq API request failed"):
        backend.classify(
            system_prompt="s",
            user_prompt="u",
            output_schema=SAMPLE_SCHEMA,
            tool_name="report_mast_classification",
        )

    # A non-transient error (e.g. bad API key) should fail immediately, not
    # burn through retries that can't possibly help.
    assert backend._client.chat.completions.create.call_count == 1
