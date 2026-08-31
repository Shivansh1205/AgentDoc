"""Tests for the MAST taxonomy reference data."""

from __future__ import annotations

import pytest

from agentdoc.classifier.taxonomy import (
    FAILURE_MODES,
    FailureCategory,
    FailureMode,
    failure_modes_by_category,
    get_failure_mode,
)


def test_exactly_14_failure_modes_defined() -> None:
    assert len(FailureMode) == 14
    assert len(FAILURE_MODES) == 14


def test_categories_have_correct_mode_counts() -> None:
    # Per the paper: FC1 has 5 modes, FC2 has 6 modes, FC3 has 3 modes.
    assert len(failure_modes_by_category(FailureCategory.SYSTEM_DESIGN)) == 5
    assert len(failure_modes_by_category(FailureCategory.INTER_AGENT_MISALIGNMENT)) == 6
    assert len(failure_modes_by_category(FailureCategory.TASK_VERIFICATION)) == 3


def test_every_failure_mode_has_nonempty_name_and_definition() -> None:
    for info in FAILURE_MODES.values():
        assert info.name.strip()
        assert info.definition.strip()


def test_get_failure_mode_by_id_string() -> None:
    info = get_failure_mode("FM-2.4")
    assert info.name == "Information withholding"
    assert info.category is FailureCategory.INTER_AGENT_MISALIGNMENT


def test_get_failure_mode_rejects_unknown_id() -> None:
    with pytest.raises(KeyError):
        get_failure_mode("FM-9.9")


def test_specific_failure_mode_names_match_paper() -> None:
    # Spot-check exact names from arXiv:2503.13657 Appendix A to guard
    # against silent renaming/drift.
    assert FAILURE_MODES[FailureMode.DISOBEY_TASK_SPECIFICATION].name == (
        "Disobey task specification"
    )
    assert FAILURE_MODES[FailureMode.STEP_REPETITION].name == "Step repetition"
    assert FAILURE_MODES[FailureMode.REASONING_ACTION_MISMATCH].name == (
        "Reasoning-action mismatch"
    )
    assert FAILURE_MODES[FailureMode.NO_OR_INCOMPLETE_VERIFICATION].name == (
        "No or incomplete verification"
    )
