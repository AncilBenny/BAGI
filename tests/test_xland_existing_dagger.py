"""Guard held-out boundaries of existing DAgger XLand integration."""
import pytest

from bagi.experiments.xland_existing_bc import TRAIN_TASKS
from bagi.experiments.xland_existing_dagger import training_task_for_episode


def test_all_training_task_ids_cycle_without_heldout():
    names = [training_task_for_episode(i) for i in range(27)]
    assert names == list(TRAIN_TASKS) * 9
    assert "hold_then_near" not in names


@pytest.mark.parametrize("episode", [-3, -1, 0.5, "6"])
def test_reject_invalid_training_episode_index(episode):
    with pytest.raises(ValueError):
        training_task_for_episode(episode)
