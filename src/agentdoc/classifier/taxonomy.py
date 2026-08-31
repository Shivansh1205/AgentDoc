"""The MAST (Multi-Agent System Failure Taxonomy) failure modes.

Source: Cemri et al., "Why Do Multi-Agent LLM Systems Fail?" (arXiv:2503.13657),
Appendix A. Names, groupings, and definitions are taken verbatim/near-verbatim
from the paper — do not rename or re-group these without re-checking the source,
since the classifier prompt and any downstream reporting depend on this being
the paper's actual taxonomy rather than an invented approximation.

MAST defines 14 failure modes (FM) across 3 failure categories (FC):
    FC1. System Design Issues       (5 modes: FM-1.1 .. FM-1.5)
    FC2. Inter-Agent Misalignment   (6 modes: FM-2.1 .. FM-2.6)
    FC3. Task Verification          (3 modes: FM-3.1 .. FM-3.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureCategory(str, Enum):
    """The 3 top-level MAST failure categories (FC1-FC3)."""

    SYSTEM_DESIGN = "system_design_issues"
    INTER_AGENT_MISALIGNMENT = "inter_agent_misalignment"
    TASK_VERIFICATION = "task_verification"


class FailureMode(str, Enum):
    """The 14 MAST failure modes (FM-1.1 .. FM-3.3), identified by paper ID."""

    # FC1. System Design Issues
    DISOBEY_TASK_SPECIFICATION = "FM-1.1"
    DISOBEY_ROLE_SPECIFICATION = "FM-1.2"
    STEP_REPETITION = "FM-1.3"
    LOSS_OF_CONVERSATION_HISTORY = "FM-1.4"
    UNAWARE_OF_TERMINATION_CONDITIONS = "FM-1.5"

    # FC2. Inter-Agent Misalignment
    CONVERSATION_RESET = "FM-2.1"
    FAIL_TO_ASK_FOR_CLARIFICATION = "FM-2.2"
    TASK_DERAILMENT = "FM-2.3"
    INFORMATION_WITHHOLDING = "FM-2.4"
    IGNORED_OTHER_AGENTS_INPUT = "FM-2.5"
    REASONING_ACTION_MISMATCH = "FM-2.6"

    # FC3. Task Verification
    PREMATURE_TERMINATION = "FM-3.1"
    NO_OR_INCOMPLETE_VERIFICATION = "FM-3.2"
    INCORRECT_VERIFICATION = "FM-3.3"


@dataclass(frozen=True)
class FailureModeInfo:
    """Static metadata for one MAST failure mode."""

    id: FailureMode
    name: str
    category: FailureCategory
    definition: str


# Definitions are the paper's Appendix A wording (Cemri et al., arXiv:2503.13657).
FAILURE_MODES: dict[FailureMode, FailureModeInfo] = {
    FailureMode.DISOBEY_TASK_SPECIFICATION: FailureModeInfo(
        id=FailureMode.DISOBEY_TASK_SPECIFICATION,
        name="Disobey task specification",
        category=FailureCategory.SYSTEM_DESIGN,
        definition=(
            "Failure to adhere to the specified constraints or requirements of "
            "a given task, leading to suboptimal or incorrect outcomes."
        ),
    ),
    FailureMode.DISOBEY_ROLE_SPECIFICATION: FailureModeInfo(
        id=FailureMode.DISOBEY_ROLE_SPECIFICATION,
        name="Disobey role specification",
        category=FailureCategory.SYSTEM_DESIGN,
        definition=(
            "Failure to adhere to the defined responsibilities and constraints "
            "of an assigned role, potentially leading to an agent behaving like "
            "another."
        ),
    ),
    FailureMode.STEP_REPETITION: FailureModeInfo(
        id=FailureMode.STEP_REPETITION,
        name="Step repetition",
        category=FailureCategory.SYSTEM_DESIGN,
        definition=(
            "Unnecessary reiteration of previously completed steps in a "
            "process, potentially causing delays or errors in task completion."
        ),
    ),
    FailureMode.LOSS_OF_CONVERSATION_HISTORY: FailureModeInfo(
        id=FailureMode.LOSS_OF_CONVERSATION_HISTORY,
        name="Loss of conversation history",
        category=FailureCategory.SYSTEM_DESIGN,
        definition=(
            "Unexpected context truncation, disregarding recent interaction "
            "history and reverting to an antecedent conversational state."
        ),
    ),
    FailureMode.UNAWARE_OF_TERMINATION_CONDITIONS: FailureModeInfo(
        id=FailureMode.UNAWARE_OF_TERMINATION_CONDITIONS,
        name="Unaware of termination conditions",
        category=FailureCategory.SYSTEM_DESIGN,
        definition=(
            "Lack of recognition or understanding of the criteria that should "
            "trigger the termination of the agents' interaction, potentially "
            "leading to unnecessary continuation."
        ),
    ),
    FailureMode.CONVERSATION_RESET: FailureModeInfo(
        id=FailureMode.CONVERSATION_RESET,
        name="Conversation reset",
        category=FailureCategory.INTER_AGENT_MISALIGNMENT,
        definition=(
            "Unexpected or unwarranted restarting of a dialogue, potentially "
            "losing context and progress made in the interaction."
        ),
    ),
    FailureMode.FAIL_TO_ASK_FOR_CLARIFICATION: FailureModeInfo(
        id=FailureMode.FAIL_TO_ASK_FOR_CLARIFICATION,
        name="Fail to ask for clarification",
        category=FailureCategory.INTER_AGENT_MISALIGNMENT,
        definition=(
            "Inability to request additional information when faced with "
            "unclear or incomplete data, potentially resulting in incorrect "
            "actions."
        ),
    ),
    FailureMode.TASK_DERAILMENT: FailureModeInfo(
        id=FailureMode.TASK_DERAILMENT,
        name="Task derailment",
        category=FailureCategory.INTER_AGENT_MISALIGNMENT,
        definition=(
            "Deviation from the intended objective or focus of a given task, "
            "potentially resulting in irrelevant or unproductive actions."
        ),
    ),
    FailureMode.INFORMATION_WITHHOLDING: FailureModeInfo(
        id=FailureMode.INFORMATION_WITHHOLDING,
        name="Information withholding",
        category=FailureCategory.INTER_AGENT_MISALIGNMENT,
        definition=(
            "Failure to share or communicate important data or insights that "
            "an agent possesses and that could impact decision-making of other "
            "agents if shared."
        ),
    ),
    FailureMode.IGNORED_OTHER_AGENTS_INPUT: FailureModeInfo(
        id=FailureMode.IGNORED_OTHER_AGENTS_INPUT,
        name="Ignored other agent's input",
        category=FailureCategory.INTER_AGENT_MISALIGNMENT,
        definition=(
            "Disregarding or failing to adequately consider input or "
            "recommendations provided by other agents in the system, "
            "potentially leading to suboptimal decisions or missed "
            "opportunities for collaboration."
        ),
    ),
    FailureMode.REASONING_ACTION_MISMATCH: FailureModeInfo(
        id=FailureMode.REASONING_ACTION_MISMATCH,
        name="Reasoning-action mismatch",
        category=FailureCategory.INTER_AGENT_MISALIGNMENT,
        definition=(
            "Discrepancy between the logical reasoning process and the actual "
            "actions taken by the agent, potentially resulting in unexpected "
            "or undesired behaviors."
        ),
    ),
    FailureMode.PREMATURE_TERMINATION: FailureModeInfo(
        id=FailureMode.PREMATURE_TERMINATION,
        name="Premature termination",
        category=FailureCategory.TASK_VERIFICATION,
        definition=(
            "Ending a dialogue, interaction or task before all necessary "
            "information has been exchanged or objectives have been met, "
            "potentially resulting in incomplete or incorrect outcomes."
        ),
    ),
    FailureMode.NO_OR_INCOMPLETE_VERIFICATION: FailureModeInfo(
        id=FailureMode.NO_OR_INCOMPLETE_VERIFICATION,
        name="No or incomplete verification",
        category=FailureCategory.TASK_VERIFICATION,
        definition=(
            "(Partial) omission of proper checking or confirmation of task "
            "outcomes or system outputs, potentially allowing errors or "
            "inconsistencies to propagate undetected."
        ),
    ),
    FailureMode.INCORRECT_VERIFICATION: FailureModeInfo(
        id=FailureMode.INCORRECT_VERIFICATION,
        name="Incorrect verification",
        category=FailureCategory.TASK_VERIFICATION,
        definition=(
            "Failure to adequately validate or cross-check crucial information "
            "or decisions during the iterations, potentially leading to errors "
            "or vulnerabilities in the system."
        ),
    ),
}

CATEGORY_NAMES: dict[FailureCategory, str] = {
    FailureCategory.SYSTEM_DESIGN: "System Design Issues",
    FailureCategory.INTER_AGENT_MISALIGNMENT: "Inter-Agent Misalignment",
    FailureCategory.TASK_VERIFICATION: "Task Verification",
}


def failure_modes_by_category(category: FailureCategory) -> list[FailureModeInfo]:
    """Return all failure modes belonging to `category`, in FM-ID order."""
    return sorted(
        (info for info in FAILURE_MODES.values() if info.category is category),
        key=lambda info: info.id.value,
    )


def get_failure_mode(failure_mode_id: str) -> FailureModeInfo:
    """Look up a failure mode by its paper ID string (e.g. "FM-2.4").

    Raises:
        KeyError: if `failure_mode_id` is not one of the 14 known MAST modes.
    """
    try:
        return FAILURE_MODES[FailureMode(failure_mode_id)]
    except ValueError as exc:
        raise KeyError(f"Unknown MAST failure mode id: {failure_mode_id!r}") from exc
