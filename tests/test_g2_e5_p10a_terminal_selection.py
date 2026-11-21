"""Synthetic, simulator-free P10a label/selection contract tests."""
from dataclasses import replace

import numpy as np
import pytest

from bagi.experiments import g2_e5_p10a_terminal_selection as p10a


def _archive(domain="xland", *, bad_label=False):
    episodes = 4
    width = 115 if domain == "xland" else 25
    obs = np.zeros((episodes * 4, width), dtype=np.int8)
    action = np.tile(np.array([0, 1, 2, 1], dtype=np.int8), episodes)
    groups = np.repeat(np.arange(episodes), 4)
    labels = np.zeros(len(action), dtype=np.int8)
    for episode in range(episodes):
        start = episode * 4
        obs[start + 1, 0] = 1
        labels[start] = 1
        labels[start + 1] = 1
        labels[start + 3] = episode % 2
    if bad_label:
        labels[0] = 0
    ids = ([f"xland-{11900 + i}-17-hold-BC" for i in range(episodes)]
           if domain == "xland" else
           [f"workshop-default-uniform_random-{12100 + i}" for i in range(episodes)])
    prefix = f"{domain}_train_"
    return {prefix + "observations": obs,
            prefix + "actions": action,
            prefix + "labels": labels,
            prefix + "groups": groups,
            prefix + "episode_ids": np.asarray(ids),
            prefix + "behaviors": np.asarray(["uniform_random"] * episodes)}


@pytest.mark.parametrize("domain", ["xland", "workshop"])
def test_balanced_train_only_swap_deterministic(domain):
    data = p10a._build_dataset(_archive(domain), domain, strict=False)
    assert len(data.action) == 16 and data.tail.sum() == 4
    assert data.observation.shape[1] == (87 if domain == "xland" else 25)
    for held_fold in (0, 1):
        one = p10a._selection(data, held_fold)
        two = p10a._selection(data, held_fold)
        assert one["removed_source_index_sha256"] == two["removed_source_index_sha256"]
        assert len(one["shifted"]) == len(one["tail_swap"]) == 6
        assert len(one["added_tails"]) == len(one["removed"]) == 2
        assert not np.intersect1d(one["held"], one["tail_swap"]).size
        assert not np.intersect1d(one["held"], one["removed"]).size
        assert np.array_equal(np.bincount(data.action[one["shifted"]], minlength=data.n_actions),
                              np.bincount(data.action[one["tail_swap"]], minlength=data.n_actions))


def test_selection_never_uses_any_label_for_removal():
    data = p10a._build_dataset(_archive(), "xland", strict=False)
    modified = replace(data, changed=1 - data.changed)
    for fold in (0, 1):
        a = p10a._selection(data, fold)
        b = p10a._selection(modified, fold)
        assert np.array_equal(a["removed"], b["removed"])
        assert a["removed_source_index_sha256"] == b["removed_source_index_sha256"]


def test_interior_direct_label_disagreement_rejected():
    with pytest.raises(ValueError, match="disagree"):
        p10a._build_dataset(_archive(bad_label=True), "xland", strict=False)


def test_insufficient_same_action_interior_support_rejected():
    data = p10a._build_dataset(_archive(), "xland", strict=False)
    action = data.action.copy()
    action[np.flatnonzero(data.tail)] = 5
    with pytest.raises(ValueError, match="Infeasible"):
        p10a._selection(replace(data, action=action), 0)


def test_shared_source_layout_stays_in_one_fold():
    archive = _archive()
    archive["xland_train_episode_ids"][1] = "xland-11900-17-hold-BC"
    data = p10a._build_dataset(archive, "xland", strict=False)
    assert np.all(data.fold[data.source_group == 11900] == 0)
    assert set(np.unique(data.fold)) == {0, 1}


def test_strict_registered_counts_fail_closed():
    with pytest.raises(ValueError, match="Registered XLand"):
        p10a._build_dataset(_archive(), "xland")
    with pytest.raises(ValueError, match="Registered random Workshop"):
        p10a._build_dataset(_archive("workshop"), "workshop")


def test_workshop_scripted_fixtures_excluded_before_selection():
    archive = _archive("workshop")
    archive["workshop_train_behaviors"][0] = "g1_public_script"
    with pytest.raises(ValueError, match="Both independent"):
        # Only one fold remains if all even-group random episodes are removed.
        archive["workshop_train_behaviors"][2] = "g1_public_script"
        p10a._build_dataset(archive, "workshop", strict=False)


def test_unregistered_domain_and_fold_rejected():
    with pytest.raises(ValueError, match="Unregistered"):
        p10a._build_dataset({}, "unknown")
    data = p10a._build_dataset(_archive(), "xland", strict=False)
    with pytest.raises(ValueError, match="Invalid held"):
        p10a._selection(data, 2)
