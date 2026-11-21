"""Stage23 matched additional GRU supervision: source replay vs real rescue suffix."""
from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from bagi.experiments.m23_executed_recovery_acquisition import (
    validate_recovery_row,
)
from bagi.experiments.m23_source_interaction_context_bc import (
    LR,
    MODEL_SEEDS,
    InteractionContextGRU,
)

ARMS = ("source_replay", "executed_rescue")
FOLLOWUP_UPDATES = 144
EPISODES_PER_STEP = 16
TOKENS_PER_UPDATE = 128


def model_sha256(model: InteractionContextGRU) -> str:
    digest = hashlib.sha256()
    for name, param in model.state_dict().items():
        digest.update(name.encode("ascii"))
        digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prepare_label_pool(
    teacher: list[dict], acquired: list[dict], seed: int, arm: str,
) -> tuple[list[tuple[dict, np.ndarray]], dict]:
    if seed not in MODEL_SEEDS or arm not in ARMS or len(teacher) != 192:
        raise ValueError("Stage23 training seed/arm/source episodes not registered")
    for row in teacher:
        validate_recovery_row(row, recovery=False)
    for row in acquired:
        validate_recovery_row(row, recovery=True)
        if row["base_model_seed"] != seed:
            raise ValueError("Stage23 recovery data from wrong frozen learner seed")
    if arm == "source_replay":
        eligible = [
            (row, np.ones(len(row["y"]), dtype=bool)) for row in teacher
        ]
    else:
        eligible = [
            (row, row["expert_executed_suffix_mask"])
            for row in acquired if row["expert_executed_suffix_mask"].any()
        ]
    unique = sum(int(mask.sum()) for _row, mask in eligible)
    if not eligible or not unique:
        raise ValueError("Stage23 no genuinely executed recovery labels available")
    return eligible, {
        "unique_source_real_supervised_tokens": unique,
        "unique_native_supervised_episodes": len(eligible),
        "labels_are_only_actually_executed_recovery_suffix": (
            arm == "executed_rescue"
        ),
    }


def fit_followup(
    teacher: list[dict], acquired: list[dict],
    frozen: InteractionContextGRU, seed: int, arm: str,
    class_weights: torch.Tensor,
) -> tuple[InteractionContextGRU, dict]:
    """Every update uses exactly 128 actual-source-label presentations."""
    if (seed not in MODEL_SEEDS or arm not in ARMS or frozen.training
            or class_weights.shape != (7,) or not torch.isfinite(class_weights).all()):
        raise ValueError("Stage23 trained public frozen GRU/weight contract invalid")
    pool, meta = prepare_label_pool(teacher, acquired, seed, arm)
    before = model_sha256(frozen)
    model = copy.deepcopy(frozen)
    if model_sha256(model) != before:
        raise ValueError("Stage23 paired frozen initialization did not match")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(22_300_000 + seed)
    labels_presented = updates = 0
    loss_sum = 0.0
    started = time.perf_counter()
    for _ in range(FOLLOWUP_UPDATES):
        ids = rng.integers(0, len(pool), size=EPISODES_PER_STEP)
        chosen = [pool[int(i)] for i in ids]
        x = nn.utils.rnn.pad_sequence(
            [torch.from_numpy(row["x"]).clone() for row, _ in chosen],
            batch_first=True,
        )
        y = nn.utils.rnn.pad_sequence(
            [torch.from_numpy(row["y"]) for row, _ in chosen],
            batch_first=True, padding_value=-100,
        )
        masks = nn.utils.rnn.pad_sequence(
            [torch.from_numpy(mask) for _, mask in chosen],
            batch_first=True, padding_value=False,
        )
        if torch.any(masks & (y == -100)):
            raise ValueError("Stage23 padded token cannot be supervised")
        valid = torch.nonzero(masks, as_tuple=False)
        if not len(valid):
            raise ValueError("Stage23 real suffix supervision batch empty")
        # Sampling WITH REPLACEMENT repeats previously executed source labels;
        # exactly 128 presentations, never new physical env.step actions.
        drawn = torch.from_numpy(
            rng.integers(0, len(valid), size=TOKENS_PER_UPDATE).astype(np.int64)
        )
        selected = valid[drawn]
        optimizer.zero_grad()
        logits = model(x)
        targets = y[selected[:, 0], selected[:, 1]]
        if torch.any((targets < 0) | (targets > 6)):
            raise ValueError("Stage23 selected label was not a native action")
        loss = nn.functional.cross_entropy(
            logits[selected[:, 0], selected[:, 1]], targets,
            weight=class_weights,
        )
        if not torch.isfinite(loss):
            raise ValueError("Stage23 nonfinite matched source/rescue loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        updates += 1
        labels_presented += len(selected)
        loss_sum += float(loss.detach())
    if (updates != FOLLOWUP_UPDATES
            or labels_presented != FOLLOWUP_UPDATES * TOKENS_PER_UPDATE
            or model_sha256(frozen) != before):
        raise ValueError("Stage23 follow-up optimizer/label budget or frozen source altered")
    model.eval()
    return model, {
        "arm": arm, "model_seed": seed,
        "frozen_base_model_sha256": before,
        "source_expert_only_initialization": True,
        "supervision_distribution": (
            "ordinary_source_teacher_replay" if arm == "source_replay"
            else "actually_executed_post_open_expert_suffix"
        ),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "real_optimizer_updates": updates,
        "episodes_per_real_update": EPISODES_PER_STEP,
        "fixed_supervised_native_labels_per_update": TOKENS_PER_UPDATE,
        "actual_cached_label_presentations_not_native_steps": labels_presented,
        "true_source_observations_not_hidden_auditor": True,
        "source_class_weights": class_weights.tolist(),
        "mean_real_supervised_loss": loss_sum / updates,
        "actual_fit_seconds": time.perf_counter() - started,
        **meta,
    }
