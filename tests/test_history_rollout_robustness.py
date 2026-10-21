"""B2: held-out observable-history memory tests."""

from bagi.experiments.history_rollout_robustness import generate, lookup_score
from bagi.worlds import WorkshopEnv


def row(split, previous, target):
    return {
        "split": split,
        "history": [[[[previous]], 0]],
        "observation": [[3]],
        "action": 1,
        "next_observation": [[target]],
        "reward": 0.0,
        "events": [],
    }


def test_history_one_can_disambiguate_heldout_trajectories():
    train = [row("train", 1, 4), row("train", 2, 5)]
    test = [row("holdout", 1, 4), row("holdout", 2, 5)]
    reactive = lookup_score(train, test, 0)
    memory = lookup_score(train, test, 1)
    assert reactive["heldout_key_coverage"] == 1.0
    assert reactive["heldout_whole_outcome_accuracy_given_seen"] == 0.5
    assert memory["heldout_whole_outcome_accuracy_given_seen"] == 1.0


def test_unseen_history_is_abstention_not_success():
    result = lookup_score([row("train", 1, 4)], [row("holdout", 9, 4)], 1)
    assert result["heldout_key_coverage"] == 0
    assert result["heldout_whole_outcome_accuracy_given_seen"] is None
    assert result["heldout_whole_outcome_accuracy_with_unknown_as_error"] == 0


def test_episode_split_and_real_interaction_accounting():
    env = WorkshopEnv(size=9, max_steps=100, view_size=5)
    design = dict(
        guided_actions=[1, 1, 3, 3, 2, 2],
        prefixes=[0, 2],
        train_episodes=3,
        holdout_episodes=2,
        continuation_steps=7,
        seed=12051,
        max_history=4,
    )
    rows, counts = generate(env, **design)
    again, same = generate(env, **design)
    assert rows == again
    assert counts == same
    assert counts["training_episodes"] == 6
    assert counts["holdout_episodes"] == 4
    assert counts["simulator_real_steps"] == (
        counts["guided_prefix_steps"] + counts["logged_continuation_steps"]
    )
    train_episodes = {r["episode"] for r in rows if r["split"] == "train"}
    val_episodes = {r["episode"] for r in rows if r["split"] == "holdout"}
    assert not (train_episodes & val_episodes)
    assert all("info" not in r and "position" not in r for r in rows)
