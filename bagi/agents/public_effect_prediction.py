"""G2-E1: standard, task-neutral public action-effect prediction contracts.

This is supervised prediction of a *visible* change, not native no-op detection,
causal skill learning, a controller, or a source of hidden-state information.
Only TRAIN episode records are passed to fit; evaluation never updates weights.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PublicEffectEpisode:
    """One actually executed episode; no evaluator-only state is stored here."""

    layout: str
    episode_id: str
    behavior: str
    observations: np.ndarray  # (transitions, current public observation width)
    actions: np.ndarray  # actually executed native actions
    visible_changes: np.ndarray  # 1 iff next public signature differs


def public_change(before: object, after: object, *, history_bits: int = 0) -> int:
    """Compare legal observations, optionally excluding *executed-history* bits.

    The history is still legal policy input; it must not make every action
    spuriously count as an environmental observation change.
    """
    left = np.asarray(before)
    right = np.asarray(after)
    if (
        left.shape != right.shape or left.size == 0
        or left.dtype.kind not in "biuf" or right.dtype.kind not in "biuf"
        or not np.isfinite(left).all() or not np.isfinite(right).all()
        or type(history_bits) is not int or history_bits < 0
        or history_bits >= left.size
    ):
        raise ValueError("Invalid pair of finite, same-shape public observations")
    x, y = left.reshape(-1), right.reshape(-1)
    if history_bits:
        x, y = x[:-history_bits], y[:-history_bits]
    return int(not np.array_equal(x, y))


def checked_episodes(
    episodes: list[PublicEffectEpisode], *, width: int, n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fail closed on invalid actions, widths, labels, and duplicate IDs."""
    if not episodes or width < 1 or n_actions < 2:
        raise ValueError("Nonempty episodes, positive width, and action space required")
    ids: set[str] = set()
    obs, actions, labels, groups = [], [], [], []
    for i, episode in enumerate(episodes):
        x = np.asarray(episode.observations)
        a = np.asarray(episode.actions)
        y = np.asarray(episode.visible_changes)
        if (
            not episode.layout or not episode.episode_id or not episode.behavior
            or episode.episode_id in ids or x.ndim != 2
            or x.shape[1] != width or x.shape[0] == 0
            or a.shape != (len(x),) or y.shape != (len(x),)
            or x.dtype.kind not in "biuf" or not np.isfinite(x).all()
            or a.dtype.kind not in "iu" or y.dtype.kind not in "biu"
            or np.any(a < 0) or np.any(a >= n_actions)
            or np.any((y != 0) & (y != 1))
        ):
            raise ValueError("Invalid public transition episode or duplicate episode ID")
        ids.add(episode.episode_id)
        obs.append(x.astype(np.float32))
        actions.append(a.astype(np.int64))
        labels.append(y.astype(np.float32))
        groups.append(np.full(len(x), i, dtype=np.int64))
    return (np.concatenate(obs), np.concatenate(actions),
            np.concatenate(labels), np.concatenate(groups))


def disjoint_layouts(train: list[PublicEffectEpisode],
                     test: list[PublicEffectEpisode]) -> None:
    if {e.layout for e in train} & {e.layout for e in test}:
        raise ValueError("TRAIN and TEST public-effect layouts overlap")
    if {e.episode_id for e in train} & {e.episode_id for e in test}:
        raise ValueError("TRAIN and TEST episode identifiers overlap")


def shuffled_training_actions(actions: np.ndarray, groups: np.ndarray,
                              *, seed: int) -> np.ndarray:
    """Shuffle actions within each TRAIN episode, retaining each histogram."""
    rng = np.random.default_rng(seed)
    shuffled = actions.copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        shuffled[indices] = actions[rng.permutation(indices)]
        if not np.array_equal(np.sort(shuffled[indices]), np.sort(actions[indices])):
            raise AssertionError("Shuffled control changed an episode action histogram")
    return shuffled


class PublicEffectLogistic(nn.Module):
    """Standard logistic regression on public observation + one-hot action."""

    def __init__(self, width: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.linear = nn.Linear(width + n_actions, 1)

    def forward(self, observations: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or actions.shape != (len(observations),):
            raise ValueError("Invalid effect predictor batch")
        onehot = torch.nn.functional.one_hot(
            actions.long(), num_classes=self.n_actions
        ).float()
        return self.linear(torch.cat((observations.float(), onehot), dim=1)).squeeze(1)


@dataclass(frozen=True)
class PublicEffectFit:
    actual_action: PublicEffectLogistic
    shuffled_action: PublicEffectLogistic
    action_frequency: np.ndarray
    constant_frequency: float
    train_rows: int
    train_episodes: int
    train_action_counts: tuple[int, ...]


def _train_logistic(model: PublicEffectLogistic, x: torch.Tensor,
                    a: torch.Tensor, y: torch.Tensor, *, seed: int,
                    epochs: int, batch_size: int, learning_rate: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(len(y), generator=generator).split(batch_size):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(x[indices], a[indices]), y[indices]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def fit_public_effects(
    train: list[PublicEffectEpisode], *, width: int, n_actions: int,
    seed: int, epochs: int = 20, batch_size: int = 64,
    learning_rate: float = 0.001,
) -> PublicEffectFit:
    """Fit two same-capacity heads with matched initialization and minibatches.

    Call with TRAIN only. Both labels must occur; do not use TEST to select
    hyperparameters, calibrate, normalize, or change early-stopping criteria.
    """
    if (
        type(seed) is not int or type(epochs) is not int or epochs < 1
        or type(batch_size) is not int or batch_size < 1
        or not np.isfinite(learning_rate) or learning_rate <= 0
    ):
        raise ValueError("Invalid registered logistic training recipe")
    obs, actions, labels, groups = checked_episodes(train, width=width,
                                                     n_actions=n_actions)
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError("Both public-change classes required in TRAIN; fail closed")
    shuffled = shuffled_training_actions(actions, groups, seed=seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        real = PublicEffectLogistic(width, n_actions)
    sham = PublicEffectLogistic(width, n_actions)
    sham.load_state_dict(real.state_dict())
    if any(not torch.equal(real.state_dict()[key], sham.state_dict()[key])
           for key in real.state_dict()):
        raise AssertionError("Control initial parameters differ")
    x = torch.from_numpy(obs)
    y = torch.from_numpy(labels)
    _train_logistic(real, x, torch.from_numpy(actions), y, seed=seed,
                    epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate)
    _train_logistic(sham, x, torch.from_numpy(shuffled), y, seed=seed,
                    epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate)
    counts = np.bincount(actions, minlength=n_actions)
    positives = np.bincount(actions, weights=labels, minlength=n_actions)
    return PublicEffectFit(real, sham, (positives + 1) / (counts + 2),
                           float(labels.mean()), len(labels), len(train),
                           tuple(int(c) for c in counts))


def effect_predictions(fit: PublicEffectFit,
                       test: list[PublicEffectEpisode], *, width: int,
                       n_actions: int) -> dict[str, np.ndarray]:
    obs, actions, _, _ = checked_episodes(test, width=width, n_actions=n_actions)
    x, a = torch.from_numpy(obs), torch.from_numpy(actions)
    fit.actual_action.eval()
    fit.shuffled_action.eval()
    with torch.no_grad():
        actual = torch.sigmoid(fit.actual_action(x, a)).numpy()
        shuffled = torch.sigmoid(fit.shuffled_action(x, a)).numpy()
    return {
        "actual_action": actual,
        "shuffled_action": shuffled,
        "action_frequency": fit.action_frequency[actions],
        "constant": np.full(len(actions), fit.constant_frequency),
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    y, p = np.asarray(labels), np.asarray(probabilities)
    if (
        y.ndim != 1 or y.shape != p.shape or len(y) == 0
        or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
        or np.any((y != 0) & (y != 1))
    ):
        raise ValueError("Invalid heldout binary prediction vectors")
    eps = 1e-7
    safe = np.clip(p, eps, 1 - eps)
    positive = p >= 0.5
    tp = int(np.count_nonzero(positive & (y == 1)))
    fp = int(np.count_nonzero(positive & (y == 0)))
    fn = int(np.count_nonzero(~positive & (y == 1)))
    ece = 0.0
    bins = np.minimum((p * 10).astype(int), 9)
    for i in range(10):
        mask = bins == i
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(p[mask]))
                                                  - float(np.mean(y[mask])))
    return {
        "rows": len(y), "prevalence": float(np.mean(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(-y * np.log(safe) - (1 - y) * np.log1p(-safe))),
        "ece_10": ece, "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def episode_bootstrap_brier_difference(
    labels: np.ndarray, first: np.ndarray, second: np.ndarray,
    groups: np.ndarray, *, seed: int, resamples: int = 1000,
) -> dict[str, float]:
    """Bootstrap whole episodes, never treat correlated steps as iid."""
    if resamples < 1 or len(labels) != len(first) or len(labels) != len(second) \
            or len(labels) != len(groups) or not len(labels):
        raise ValueError("Invalid episode bootstrap inputs")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two heldout episodes required for bootstrap")
    per_episode = [np.flatnonzero(groups == i) for i in unique]
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for i in range(resamples):
        sample = np.concatenate([per_episode[j] for j in rng.integers(
            0, len(unique), size=len(unique)
        )])
        differences[i] = float(np.mean((first[sample] - labels[sample]) ** 2)
                               - np.mean((second[sample] - labels[sample]) ** 2))
    return {"difference_first_minus_second": float(np.mean((first - labels) ** 2)
                                                   - np.mean((second - labels) ** 2)),
            "ci_95_low": float(np.quantile(differences, 0.025)),
            "ci_95_high": float(np.quantile(differences, 0.975))}


def duplicate_trajectory_groups(episodes: list[PublicEffectEpisode]) -> dict[str, int]:
    """Detect identical trajectories even across different episode identifiers."""
    fingerprints = Counter()
    for episode in episodes:
        x = np.ascontiguousarray(np.asarray(episode.observations, dtype=np.float32))
        a = np.ascontiguousarray(np.asarray(episode.actions, dtype=np.int64))
        y = np.ascontiguousarray(np.asarray(episode.visible_changes, dtype=np.int8))
        payload = (str(x.shape).encode() + x.tobytes() + a.tobytes() + y.tobytes())
        fingerprints[hashlib.sha256(payload).hexdigest()] += 1
    return {"episodes": len(episodes), "unique_trajectories": len(fingerprints),
            "duplicate_episodes": sum(n - 1 for n in fingerprints.values())}
