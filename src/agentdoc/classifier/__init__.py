"""MAST classification engine.

Classifies a `NormalizedTrace` (see `agentdoc.parsers.schema`) against the 14
MAST failure modes across the 3 top-level categories: system-design issues,
inter-agent misalignment, and task-verification failures. See
`agentdoc.classifier.taxonomy` for the taxonomy itself and
`agentdoc.classifier.engine` for the LLM-as-a-judge classifier.
"""

from agentdoc.classifier.engine import BACKENDS, DEFAULT_BACKEND, MastClassifier, build_backend
from agentdoc.classifier.llm import (
    AnthropicBackend,
    GroqBackend,
    LLMBackend,
    LLMBackendError,
)
from agentdoc.classifier.results import ClassificationResult, FlaggedFailure
from agentdoc.classifier.taxonomy import (
    FAILURE_MODES,
    FailureCategory,
    FailureMode,
    FailureModeInfo,
)

__all__ = [
    "MastClassifier",
    "build_backend",
    "BACKENDS",
    "DEFAULT_BACKEND",
    "LLMBackend",
    "AnthropicBackend",
    "GroqBackend",
    "LLMBackendError",
    "ClassificationResult",
    "FlaggedFailure",
    "FailureCategory",
    "FailureMode",
    "FailureModeInfo",
    "FAILURE_MODES",
]
