"""M23 source-only acquisition rules; these do not establish a learned skill."""

import pytest

from bagi.experiments.m23_source_feasibility import (
    SOURCE_ENV_IDS,
    PublicObjectExplorer,
    run_source_episode,
)
from bagi.experiments.m23_study_contract import TRAIN_LAYOUTS


def test_source_acquisition_contains_no_composition_target():
    assert set(SOURCE_ENV_IDS) == {"doorkey", "fetch", "putnear"}
    assert all(len(TRAIN_LAYOUTS[name]) == 128 for name in SOURCE_ENV_IDS)
    assert "unlockpickup" not in SOURCE_ENV_IDS


def test_rejects_final_layout_before_any_native_step():
    with pytest.raises(ValueError, match="TRAIN"):
        run_source_episode("fetch", 443001)
    with pytest.raises(ValueError, match="TRAIN"):
        run_source_episode("doorkey", 433001)
    with pytest.raises(ValueError, match="TRAIN"):
        run_source_episode("unlockpickup", 451001)


def test_explorer_memory_starts_from_only_executed_evidence():
    actor = PublicObjectExplorer()
    assert not actor.public_carrying_evidence
