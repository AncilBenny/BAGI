"""Stage20 equal-label public recurrent correction on shared real on-policy trajectories.

Two final arms see identical native executed source episodes, frozen-learner
visited states and public expert ADVISORY labels. Their only intervention is
WHICH source-actually-visited advice tokens receive a capped 4x CE weight.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from bagi.experiments.m23_post_open_acquisition import TASKS
from bagi.experiments.m23_source_interaction_context_bc import (
    EPISODE_BATCH,
    EPOCHS,
    LR,
    MODEL_SEEDS,
    WIDTH,
    InteractionContextGRU,
    interaction_context_weights,
)

ARMS = ("uniform_corrected", "post_open_corrected")
FACTOR = 4.0


def acquired_token_masks(
    episodes: list[dict], seed: int
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Select exact same boost count but distinct temporal placement per task."""
    if seed not in MODEL_SEEDS or not episodes:
        raise ValueError("Stage20 acquired public advice masks require registered seed")
    treatment = [np.zeros(len(r["y"]), dtype=bool) for r in episodes]
    control = [np.zeros(len(r["y"]), dtype=bool) for r in episodes]
    counts: dict[str, dict] = {}
    for task_index, task in enumerate(TASKS):
        indices = [i for i, r in enumerate(episodes) if r["task"] == task]
        if not indices:
            raise ValueError("Stage20 missing acquired native source task")
        for i in indices:
            row = episodes[i]
            opened = row["public_first_witnessed_open_action_index"]
            executed = row["actually_executed"]
            if (len(row["y"]) != len(executed) or len(row["y"]) != len(row["x"])
                    or len(row["y"]) == 0):
                raise ValueError("Stage20 real source advice/history length mismatch")
            if opened is not None:
                if (task != "doorkey" or type(opened) is not int
                        or not 0 <= opened < len(executed)
                        or int(executed[opened]) != 5):
                    raise ValueError("Stage20 impossible learner public door witness")
                treatment[i][opened + 1:] = True
        post_count = sum(int(treatment[i].sum()) for i in indices)
        total = sum(len(episodes[i]["y"]) for i in indices)
        if not 0 <= post_count <= total:
            raise ValueError("Stage20 post-open public advice mask out of bounds")
        # The control uses the SAME unique, real acquired labels and the SAME
        # count of boosted tokens, sampled without replacement, fixed before
        # any independent MONITOR source physical actions.
        rng = np.random.default_rng(20_000 + seed + task_index * 1_000_000)
        selected = np.zeros(total, dtype=bool)
        if post_count:
            selected[rng.choice(total, size=post_count, replace=False)] = True
        offset = 0
        for i in indices:
            n = len(episodes[i]["y"])
            control[i][:] = selected[offset:offset + n]
            offset += n
        if offset != total:
            raise ValueError("Stage20 real acquired labels partition mismatch")
        if sum(int(control[i].sum()) for i in indices) != post_count:
            raise ValueError("Stage20 boosted advice label counts not matched")
        counts[task] = {
            "actual_source_acquired_label_count": total,
            "public_post_open_true_teacher_advice_labels": post_count,
            "control_deterministically_boosted_true_advice_labels": post_count,
            "treatment_true_post_open_boosted_advice_labels": post_count,
        }
    if sum(len(r["y"]) for r in episodes) != 2 * 6144:
        raise ValueError("Stage20 shared learner acquisition action budget changed")
    return control, treatment, counts


def validate_public_train_episode(row: dict, acquisition: bool) -> None:
    """Never confuse executed learner action history with teacher advice labels."""
    x, y = row["x"], row["y"]
    actual = row["actually_executed"]
    if (row["task"] not in TASKS or x.ndim != 2 or x.shape[1] != WIDTH
            or y.ndim != 1 or actual.ndim != 1
            or not len(y) or len(x) != len(y) or len(actual) != len(y)
            or len(y) > 256 or not np.isfinite(x).all()
            or set(np.unique(y)) - set(range(7))
            or set(np.unique(actual)) - set(range(7))):
        raise ValueError("Stage20 real public state/teacher-advice alignment invalid")
    if not np.array_equal(x[0, -7:], np.zeros(7, dtype=np.float32)):
        raise ValueError("Stage20 recurrent history not reset per actual episode")
    if len(y) > 1 and not np.array_equal(
        x[1:, -7:], np.eye(7, dtype=np.float32)[actual[:-1]]
    ):
        raise ValueError("Stage20 previous native action not actually executed")
    if not acquisition and not np.array_equal(y, actual):
        raise ValueError("Stage20 original expert source label was not executed")
    if acquisition and row["origin"] != "frozen_learner_public_advice":
        raise ValueError("Stage20 acquisition must be frozen real learner visits")


def fit_corrected(
    teacher: list[dict], acquired: list[dict], model_seed: int,
    arm: str, source_class_weights: torch.Tensor,
) -> tuple[InteractionContextGRU, dict]:
    if (model_seed not in MODEL_SEEDS or arm not in ARMS
            or len(teacher) != 192 or not acquired
            or source_class_weights.shape != (7,)
            or not torch.isfinite(source_class_weights).all()):
        raise ValueError("Stage20 final GRU training setup unregistered")
    for row in teacher:
        validate_public_train_episode(row, acquisition=False)
    for row in acquired:
        validate_public_train_episode(row, acquisition=True)
        if row["base_model_seed"] != model_seed:
            raise ValueError("Stage20 mixed up frozen seed visitation histories")
    control, post, masks = acquired_token_masks(acquired, model_seed)
    applied_masks = control if arm == "uniform_corrected" else post
    rows = teacher + acquired
    scales = [
        interaction_context_weights(row["y"]) for row in teacher
    ] + [
        np.where(mask, FACTOR, 1.0).astype(np.float32)
        for mask in applied_masks
    ]
    if len(rows) != len(scales) or any(
        len(row["y"]) != len(scale)
        for row, scale in zip(rows, scales, strict=True)
    ):
        raise ValueError("Stage20 true source label weights misaligned")
    total_tokens = sum(len(row["y"]) for row in rows)
    torch.manual_seed(model_seed)
    model = InteractionContextGRU()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    shuffle = torch.Generator().manual_seed(10_000 + model_seed)
    steps = presentations = 0
    loss_sum = 0.0
    start = time.perf_counter()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(rows), generator=shuffle).split(EPISODE_BATCH):
            ids = [int(i) for i in idx]
            x = nn.utils.rnn.pad_sequence(
                [torch.from_numpy(rows[i]["x"]).clone() for i in ids],
                batch_first=True,
            )
            targets = nn.utils.rnn.pad_sequence(
                [torch.from_numpy(rows[i]["y"]) for i in ids],
                batch_first=True, padding_value=-100,
            )
            weights = nn.utils.rnn.pad_sequence(
                [torch.from_numpy(scales[i]) for i in ids],
                batch_first=True, padding_value=0.0,
            )
            valid = targets != -100
            if int(valid.sum().item()) != sum(len(rows[i]["y"]) for i in ids):
                raise ValueError("Stage20 native source and advice padding invalid")
            optimizer.zero_grad()
            logits = model(x)
            token_loss = nn.functional.cross_entropy(
                logits[valid], targets[valid],
                weight=source_class_weights, reduction="none",
            )
            weight = weights[valid]
            normalizer = (source_class_weights[targets[valid]] * weight).sum()
            loss = (token_loss * weight).sum() / normalizer
            if not torch.isfinite(loss):
                raise ValueError("Stage20 nonfinite matched public advice loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            presentations += int(valid.sum().item())
            loss_sum += float(loss.detach())
    if (steps != EPOCHS * ((len(rows) + EPISODE_BATCH - 1) // EPISODE_BATCH)
            or presentations != EPOCHS * total_tokens):
        raise ValueError("Stage20 equal source/advice fit-exposure accounting failed")
    model.eval()
    return model, {
        "arm": arm,
        "model_seed": model_seed,
        "architecture": "GRU(full_public_1056,hidden64,batch_first) Linear(64,7)",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "real_original_teacher_train_episodes": len(teacher),
        "shared_actual_learner_acquisition_episode_count": len(acquired),
        "real_original_teacher_executed_label_count": sum(len(r["y"]) for r in teacher),
        "actual_acquired_learner_visited_advice_label_count":
            sum(len(r["y"]) for r in acquired),
        "public_advice_4x_boost_mask_counts": masks,
        "initialization": "fresh_same_model_seed_both_final_arms",
        "epoch_count": EPOCHS,
        "episode_minibatch": EPISODE_BATCH,
        "adam_learning_rate": LR,
        "native_class_weight_source": "original_genuinely_executed_source_teacher_only",
        "source_class_weights": source_class_weights.tolist(),
        "actual_optimizer_step_calls": steps,
        "cached_label_presentations_not_new_native_steps": presentations,
        "mean_native_supervised_loss": loss_sum / steps,
        "actual_fit_seconds": time.perf_counter() - start,
    }
