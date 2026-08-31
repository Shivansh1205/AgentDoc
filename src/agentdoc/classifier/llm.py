"""Pluggable LLM backends for the MAST classifier.

The classifier engine only depends on the `LLMBackend` protocol below, not on
any specific vendor SDK. Two backends are implemented:

- `AnthropicBackend`, using Claude's tool-use (function calling) feature to
  force structured JSON output rather than parsing free text.
- `GroqBackend`, using Groq's OpenAI-compatible chat completions API with
  strict JSON Schema structured outputs (`response_format={"type":
  "json_schema", ...}`). Groq does not support combining `response_format`
  with tool use in one request, so this backend uses schema mode rather than
  tool-calling — both mechanisms achieve the same goal (schema-conformant
  output, no free-text parsing), they're just different vendor features.

Adding another provider later means implementing this same protocol against
its own structured-output mechanism — the engine and prompt-building code do
not need to change.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable


class LLMBackendError(RuntimeError):
    """Raised when an LLM backend fails to produce a usable structured response."""


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface the classifier engine needs from an LLM provider.

    A backend's job is narrow: given a system prompt, a user prompt, and a
    JSON schema describing the desired output shape, return a dict that
    conforms to that schema. All prompt construction and response
    interpretation happens in `agentdoc.classifier.engine` and
    `agentdoc.classifier.prompts` — backends should not need MAST-specific
    knowledge.
    """

    #: Identifier for provenance, e.g. "claude-sonnet-5" or "gpt-4o".
    model: str

    def classify(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        """Run one structured-output completion.

        Args:
            system_prompt: The system/instructions prompt.
            user_prompt: The user-turn content (trace + task description).
            output_schema: A JSON Schema object describing the required
                shape of the returned dict.
            tool_name: Name to give the structured-output tool/function, for
                backends that implement this via tool-calling.

        Returns:
            A dict conforming to `output_schema`.

        Raises:
            LLMBackendError: if the backend cannot produce a conforming
                structured response (API error, missing/invalid tool call, etc).
        """
        ...


class AnthropicBackend:
    """LLM backend using Anthropic's Messages API via the `anthropic` SDK.

    Structured output is obtained via forced tool use: we hand the model a
    single tool whose input schema is `output_schema`, and require
    `tool_choice={"type": "tool", "name": tool_name}`, so the model's
    response is always a validated tool call rather than free text we'd have
    to parse ourselves.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise LLMBackendError(
                "No Anthropic API key found. Get one at "
                "https://console.anthropic.com/settings/keys, then set it as "
                "the ANTHROPIC_API_KEY environment variable (or add it to a "
                ".env file in the current directory)."
            )

        try:
            import anthropic
        except ImportError as exc:
            raise LLMBackendError(
                "The 'anthropic' package is required to use AnthropicBackend. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=self._api_key)

    def classify(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        import anthropic

        tool = {
            "name": tool_name,
            "description": (
                "Report the MAST failure mode classification for the given trace."
            ),
            "input_schema": output_schema,
        }

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise LLMBackendError(f"Anthropic API request failed: {exc}") from exc

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input

        raise LLMBackendError(
            "Anthropic response did not contain the expected tool call "
            f"({tool_name!r}). Got content blocks: "
            f"{[block.type for block in response.content]!r}"
        )


class GroqBackend:
    """LLM backend using Groq's chat completions API via the `groq` SDK.

    Defaults to `openai/gpt-oss-120b`: as of this writing it's Groq's
    strongest model that supports *strict* JSON Schema structured outputs
    (`response_format={"type": "json_schema", "json_schema": {"strict": True,
    ...}}`), and it's available on Groq's free tier. Strict mode makes the
    API itself reject non-conforming output, so — like the Anthropic
    backend's forced tool use — this never falls back to parsing free text.

    Note: Groq does not currently support combining `response_format` with
    `tools`/`tool_choice` in the same request, so this backend uses schema
    mode instead of tool-calling (unlike `AnthropicBackend`). The public
    `classify()` contract is identical either way.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise LLMBackendError(
                "No Groq API key found. Get a free key at "
                "https://console.groq.com/keys, then set it as the "
                "GROQ_API_KEY environment variable (or add it to a .env "
                "file in the current directory)."
            )

        try:
            import groq
        except ImportError as exc:
            raise LLMBackendError(
                "The 'groq' package is required to use GroqBackend. "
                "Install it with: pip install groq"
            ) from exc

        self._client = groq.Groq(api_key=self._api_key)

    #: Max attempts for a Groq generation that fails its own schema
    #: validation (`json_validate_failed`) — an intermittent issue where the
    #: model emits a malformed value (e.g. "0. nine" for a number) that Groq
    #: itself rejects server-side before ever returning content to parse.
    #: Retrying is worthwhile here specifically because it's a transient
    #: sampling issue, not a genuine schema mismatch on our end.
    _MAX_GENERATION_ATTEMPTS = 3

    def classify(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        import groq

        # `tool_name` doubles as the schema's name — Groq's json_schema mode
        # requires one, mirroring how the Anthropic backend names its tool.
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": tool_name,
                "strict": True,
                "schema": output_schema,
            },
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self._MAX_GENERATION_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=self.max_tokens,
                    response_format=response_format,
                    messages=messages,
                )
            except groq.APIError as exc:
                if _is_transient_generation_failure(exc) and attempt < (
                    self._MAX_GENERATION_ATTEMPTS
                ):
                    last_error = exc
                    continue
                raise LLMBackendError(f"Groq API request failed: {exc}") from exc

            choice = response.choices[0] if response.choices else None
            content = choice.message.content if choice and choice.message else None
            if not content:
                raise LLMBackendError(
                    "Groq response did not contain any message content to "
                    "parse as structured JSON."
                )

            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise LLMBackendError(
                    f"Groq response content was not valid JSON despite "
                    f"strict json_schema mode: {content!r}"
                ) from exc

        # Unreachable in practice (the loop always returns or raises), but
        # keeps type checkers happy and fails loudly if that ever changes.
        raise LLMBackendError(
            f"Groq API request failed after {self._MAX_GENERATION_ATTEMPTS} "
            f"attempts: {last_error}"
        )


def _is_transient_generation_failure(exc: Exception) -> bool:
    """Whether a Groq API error looks like a retryable bad-generation blip.

    Specifically Groq's `json_validate_failed` error: the model produced a
    malformed value that failed strict schema validation server-side. This
    is a sampling issue, not a mismatch between our schema and the response
    shape, so retrying the same request is a reasonable recovery — unlike
    other 4xx errors (bad API key, invalid model) which won't be fixed by
    trying again.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get("error", {}).get("code") == "json_validate_failed":
            return True
    return "json_validate_failed" in str(exc)
