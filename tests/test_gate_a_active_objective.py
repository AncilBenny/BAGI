"""Check registered four-arm contrasts, stop rule and native runner boundaries."""
import pytest

from bagi.experiments.gate_a_active_objective import (
    ARMS,
    BUDGET,
    INTERVAL,
    arm_metrics,
    summarize,
    train_interval,
)
from bagi.experiments.gate_a_sequential_pilot import DIGESTS


def curves():
    rows = []
    for seed in DIGESTS:
        for arm in ARMS:
            active = arm.startswith("active_")
            curve = []
            for i, steps in enumerate(range(0, BUDGET + 1, INTERVAL)):
                values = {
                    "A_greedy": 0.7 if i == 2 else 1.0,
                    "A_live": 0.9,
                    "B_greedy": 0.0 if i == 0 else (1.0 if active else 0.6),
                    "B_live": 0.0 if i == 0 else (0.8 if active else 0.6),
                }
                curve.append({"B_actions": steps, **{k: {"success": v} for k, v in values.items()}})
            rows.append({"seed": seed, "arm": arm, "curve": curve})
    return rows


def test_factorial_pilot_threshold_and_interim_forgetting():
    rows = curves()
    summary = summarize(rows)
    assert summary["qualifying_adam_conditions"] == ["carry", "reset"]
    assert summary["per_seed"][0]["carry"]["active"]["A_greedy_worst_loss"] == pytest.approx(0.3)
    assert summary["per_seed"][0]["carry"]["active"]["A_greedy_final_loss"] == 0
    assert summary["spread"]["carry"]["B_greedy_auc"]["mean"] == pytest.approx(0.35)
    assert summary["spread"]["interaction_reset_minus_carry"]["B_greedy_auc"]["mean"] == 0
    for row in rows:
        if row["seed"] == 131 and row["arm"].startswith("active_"):
            row["curve"][-1]["A_live"]["success"] = 0.7
    assert not summarize(rows)["pilot_threshold_met"]
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(rows[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(rows + rows[:1])
    with pytest.raises(ValueError, match="measurement grid"):
        arm_metrics(rows[0]["curve"][:-1])


@pytest.mark.parametrize("arm", ["joint_carry", "active_reset"])
def test_native_interval_drains_and_tracks_cursor(arm):
    pytest.importorskip("sb3_contrib")
    import torch

    from bagi.agents.plasticity import reset_adam_history
    from bagi.experiments.cumulative_transfer import TASKS
    from bagi.experiments.gate_a_competence import BLOCK
    from bagi.experiments.gate_a_shared_interface import SharedPPO

    torch.set_num_threads(1)
    c = SharedPPO(TASKS[0], 227, 99901)
    try:
        c.train(BLOCK)
        c.env.rows.clear()
        if arm.endswith("_reset"):
            assert reset_adam_history(c.model.policy)
        rows, pending = train_interval(c, arm, 99911, BLOCK)
        assert rows[0]["seed"] == 99911
        rows, _ = train_interval(c, arm, pending, BLOCK)
        assert rows[0]["seed"] == pending
        assert c.training_actions == 3 * BLOCK
        assert c.updates == 48 and c.presentations == 12 * BLOCK
    finally:
        c.model_env.close()
