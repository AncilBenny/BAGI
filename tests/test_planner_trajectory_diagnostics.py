import numpy as np

from bagi.experiments.planner_trajectory_diagnostics import (
    merge_metrics,
    new_metrics,
    summarize_metrics,
    update_metrics,
)


def test_metric_accumulation_and_summary():
    first = new_metrics()
    update_metrics(
        first,
        predicted_next=np.asarray([[1, 2], [3, 4]]),
        actual_next=np.asarray([[1, 0], [3, 4]]),
        predicted_reward=0.25,
        actual_reward=1.0,
        event_probs=np.asarray([0.9, 0.1, 0.2, 0.8], dtype=np.float32),
        event_targets=np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    second = new_metrics()
    merge_metrics(second, first)
    summary = summarize_metrics(second)

    assert summary["steps"] == 1
    assert summary["cell_accuracy"] == 0.75
    assert summary["reward_mae"] == 0.75
    assert 0.0 <= summary["event_brier"] <= 1.0
    assert summary["event_step_brier"] == summary["event_brier"]
    assert summary["non_event_step_brier"] == 0.0
