"""Real P8/P10a scoring integration with synthetic TRAIN data and ZERO updates."""

import numpy as np

from bagi.experiments import g2_e5_p8_public_effect as p8
from bagi.experiments import g2_e5_p10a_offline_scoring as p10a_score


def _synthetic_train(domain: str) -> dict:
    episodes = 4
    width = 115 if domain == "xland" else 25
    obs = np.zeros((episodes * 4, width), dtype=np.int8)
    actions = np.tile(np.array([0, 1, 2, 1], dtype=np.int8), episodes)
    groups = np.repeat(np.arange(episodes), 4)
    labels = np.zeros(len(actions), dtype=np.int8)
    for episode in range(episodes):
        obs[episode * 4 + 1, 0] = 1
        labels[episode * 4] = 1
        labels[episode * 4 + 1] = 1
        labels[episode * 4 + 3] = episode % 2
    if domain == "xland":
        ids = [f"xland-{11900 + i}-17-hold-BC" for i in range(episodes)]
    else:
        ids = [f"workshop-default-uniform_random-{12100 + i}" for i in range(episodes)]
    prefix = f"{domain}_train_"
    return {prefix + "observations": obs, prefix + "actions": actions,
            prefix + "groups": groups, prefix + "labels": labels,
            prefix + "episode_ids": np.asarray(ids),
            prefix + "behaviors": np.asarray(["uniform_random"] * episodes)}


def test_actual_p8_model_and_p10a_scoring_without_training(monkeypatch):
    """Exercise the production 48-arm dispatch, but forbid all gradient updates."""
    monkeypatch.setattr(p8, "EPOCHS", 0)

    def forbid_step(*_args, **_kwargs):
        raise AssertionError("A synthetic scorer integration test must never optimize")

    monkeypatch.setattr(p8.torch.optim.Adam, "step", forbid_step)
    archive = {**_synthetic_train("xland"), **_synthetic_train("workshop")}
    result = p10a_score.score_archive(archive, strict=False)
    # The scorer dispatches 48 neural arms; zero epochs means ZERO model fits.
    assert result["completed_fits"] == 48
    assert result["supervised_optimizer_steps"] == 0
    assert result["native_actions"] == result["native_resets"] == 0
    assert result["original_test_eval_array_reads"] == 0
    assert len(result["records"]) == 96
    assert {record["dataset"] for record in result["records"]} == {
        "e1_xland_train", "e1_workshop_train"}
    for record in result["records"]:
        assert record["train_rows"] == 6
        assert record["held_rows"] == 8
        for subset in ("all", "non_tail", "tail"):
            metrics = record["subsets"][subset]["metrics"]
            assert metrics is not None
            assert np.isfinite(metrics["brier"])
            assert 0 <= metrics["brier"] <= 1
    for domain in ("e1_xland_train", "e1_workshop_train"):
        for contract in p10a_score.CONTRACTS:
            for arm in (*p8.ARMS, *p8.BASELINES):
                assert result["group_summary"][domain][contract][arm]["all"]["n_groups"] == 4
                assert np.isfinite(result["pooled_ece"][domain][contract][arm])
