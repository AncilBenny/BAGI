"""M23 source-only public-action dynamics fit and independent source-monitor test.

No target composition environment is opened. Reuses verified source TRAIN public
transitions and trains fresh PyTorch models; no hidden evaluator state is input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m5_continual_retention import mission_vector
from bagi.experiments.m23_source_feasibility import SOURCE_ENV_IDS, run_source_episode
from bagi.experiments.m23_study_contract import (
    MODEL_SEEDS,
    MONITOR_LAYOUTS,
    validate_split_contract,
)

TRAIN_SOURCE_COMMIT = "594c670a0d501fbd3cafe78ef47d59bef4733c0a"
SOURCE_ARTIFACT_ID = 14127428720
SOURCE_ARTIFACT_SHA256 = "dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb"
OUT = "runs/m23_world_model"
N_OBJECT = len(OBJECT_TO_IDX)
N_COLOR = len(COLOR_TO_IDX)
N_STATE = len(STATE_TO_IDX)
HISTORY = 4
N_ACTIONS = 7
BATCH_SIZE = 256
EPOCHS = 6
LR = 0.001
N_CELLS = 49
WIDTH = 7 * 7 * 3 + 4 + 64 + HISTORY * 8 + N_ACTIONS


class PublicNextFramePredictor(nn.Module):
    """Small reusable action-conditioned public-state prediction baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(WIDTH, 192), nn.ReLU(), nn.Linear(192, 192), nn.ReLU()
        )
        self.obj_head = nn.Linear(192, N_CELLS * N_OBJECT)
        self.color_head = nn.Linear(192, N_CELLS * N_COLOR)
        self.state_head = nn.Linear(192, N_CELLS * N_STATE)
        self.direction_head = nn.Linear(192, 4)
        self.reward_head = nn.Linear(192, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.backbone(x)
        return (
            self.obj_head(h).reshape(-1, N_CELLS, N_OBJECT),
            self.color_head(h).reshape(-1, N_CELLS, N_COLOR),
            self.state_head(h).reshape(-1, N_CELLS, N_STATE),
            self.direction_head(h),
            self.reward_head(h).flatten(),
        )


def public_features(image: np.ndarray, direction: int, mission: str,
                    executed_action: int, prior_actions: tuple[int, ...]) -> np.ndarray:
    """Only local public pixels, public mission and OWN executed action history."""
    image = np.asarray(image)
    if image.shape != (7, 7, 3) or len(prior_actions) != HISTORY:
        raise ValueError("M23 public observation/history contract changed")
    if not (0 <= direction < 4 and 0 <= executed_action < N_ACTIONS):
        raise ValueError("M23 invalid native direction/action")
    if any(not 0 <= a <= N_ACTIONS for a in prior_actions):
        raise ValueError("M23 history must contain executed actions or padding")
    for k, count in enumerate((N_OBJECT, N_COLOR, N_STATE)):
        if np.any(image[:, :, k] >= count) or np.any(image[:, :, k] < 0):
            raise ValueError("M23 invalid public image categorical value")
    image_feat = image.astype(np.float32).copy()
    image_feat[:, :, 0] /= max(1, N_OBJECT - 1)
    image_feat[:, :, 1] /= max(1, N_COLOR - 1)
    image_feat[:, :, 2] /= max(1, N_STATE - 1)
    d = np.eye(4, dtype=np.float32)[direction]
    history = np.eye(N_ACTIONS + 1, dtype=np.float32)[list(prior_actions)].ravel()
    action = np.eye(N_ACTIONS, dtype=np.float32)[executed_action]
    result = np.concatenate((image_feat.flatten(), d, mission_vector(mission),
                             history, action)).astype(np.float32)
    if result.shape != (WIDTH,):
        raise ValueError("M23 predictor feature width mismatch")
    return result


def encode_public_rows(rows: dict) -> dict[str, np.ndarray]:
    """Build batches without leaking future actions across episode boundaries."""
    required = ("observation_image", "observation_direction", "mission", "action",
                "next_image", "next_direction", "reward", "terminated", "truncated")
    n = len(rows["action"])
    if n == 0 or any(len(rows[k]) != n for k in required):
        raise ValueError("M23 incomplete source public transitions")
    features = []
    history = deque([N_ACTIONS] * HISTORY, maxlen=HISTORY)
    for i in range(n):
        a = int(rows["action"][i])
        features.append(public_features(
            rows["observation_image"][i], int(rows["observation_direction"][i]),
            str(rows["mission"][i]), a, tuple(history),
        ))
        history.append(a)
        if bool(rows["terminated"][i]) or bool(rows["truncated"][i]):
            history = deque([N_ACTIONS] * HISTORY, maxlen=HISTORY)
    image = np.asarray(rows["next_image"], dtype=np.int64)
    current = np.asarray(rows["observation_image"], dtype=np.int64)
    if image.shape != (n, 7, 7, 3) or current.shape != image.shape:
        raise ValueError("M23 public image batch contract changed")
    if not np.isfinite(np.asarray(rows["reward"], dtype=np.float32)).all():
        raise ValueError("M23 public reward not finite")
    return {
        "features": np.asarray(features, dtype=np.float32),
        "next_obj": image[:, :, :, 0].reshape(n, -1),
        "next_color": image[:, :, :, 1].reshape(n, -1),
        "next_state": image[:, :, :, 2].reshape(n, -1),
        "next_direction": np.asarray(rows["next_direction"], dtype=np.int64),
        "next_reward": np.asarray(rows["reward"], dtype=np.float32),
        "prior_obj": current[:, :, :, 0].reshape(n, -1),
        "action": np.asarray(rows["action"], dtype=np.int64),
    }


def _find_unique(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise ValueError(f"M23 requires exactly one verified artifact member {name}")
    return hits[0]


def load_source_artifact(root: Path) -> tuple[dict, dict[str, dict]]:
    marker = _find_unique(root, "research_branch_commit.txt")
    if marker.read_text(encoding="utf-8").strip() != TRAIN_SOURCE_COMMIT:
        raise ValueError("M23 cached source marker mismatch")
    mode = _find_unique(root, "study_mode.txt")
    if mode.read_text(encoding="utf-8").strip() != "m23":
        raise ValueError("M23 source artifact is not original source acquisition")
    report_path = _find_unique(root, "m23_compositional_transfer-summary.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["source_commit"] != TRAIN_SOURCE_COMMIT or not report["source_mechanism_coverage_pass"]:
        raise ValueError("M23 source physical mechanism gate was not met")
    out = {}
    for task in SOURCE_ENV_IDS:
        path = _find_unique(root, f"{task}-public-train-only.npz")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != report["source_tasks"][task]["public_data_sha256"]:
            raise ValueError("M23 source public dataset hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            out[task] = {k: data[k].copy() for k in data.files}
        if len(out[task]["action"]) != report["source_tasks"][task]["native_actions"]:
            raise ValueError("M23 source native actions do not match public dataset")
    if sum(len(x["action"]) for x in out.values()) != report["actual_native_actions"]:
        raise ValueError("M23 total cached source actions do not match original run")
    return report, out


def combine(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("Empty source public dataset")
    return {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}


def source_monitor_data() -> tuple[dict[str, np.ndarray], dict]:
    """Executes only new source MONITOR layouts, never a target environment."""
    observations = []
    summary = {}
    for task in SOURCE_ENV_IDS:
        rows_per_key = {}
        count = 0
        effect_counts = {}
        for seed in MONITOR_LAYOUTS[task]:
            scored, rows = run_source_episode(task, seed, split="monitor")
            count += scored["native_actions"]
            for name, n in scored["events"].items():
                effect_counts[name] = effect_counts.get(name, 0) + n
            for key, values in rows.items():
                rows_per_key.setdefault(key, []).extend(values)
        summary[task] = {"monitor_episodes": len(MONITOR_LAYOUTS[task]),
                         "actual_native_actions": count, "physical_effects": effect_counts}
        observations.append(encode_public_rows(rows_per_key))
    return combine(observations), summary


def _torch_data(rows: dict[str, np.ndarray]):
    return {k: torch.from_numpy(v.copy()).to(torch.float32 if k in
            ("features", "next_reward") else torch.int64) for k, v in rows.items()
            if k not in ("prior_obj", "action")}


def fit_one(rows: dict[str, np.ndarray], seed: int) -> tuple[PublicNextFramePredictor, dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = PublicNextFramePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data = _torch_data(rows)
    n = len(data["features"])
    updates = 0
    last_loss = None
    model.train()
    for _ in range(EPOCHS):
        shuffled = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = torch.as_tensor(shuffled[start:start + BATCH_SIZE])
            logits = model(data["features"][idx])
            losses = (
                torch.nn.functional.cross_entropy(logits[0].flatten(0, 1),
                                                  data["next_obj"][idx].flatten()),
                torch.nn.functional.cross_entropy(logits[1].flatten(0, 1),
                                                  data["next_color"][idx].flatten()),
                torch.nn.functional.cross_entropy(logits[2].flatten(0, 1),
                                                  data["next_state"][idx].flatten()),
                torch.nn.functional.cross_entropy(logits[3], data["next_direction"][idx]),
                torch.nn.functional.mse_loss(logits[4], data["next_reward"][idx]),
            )
            loss = losses[0] + losses[1] + losses[2] + 0.2 * losses[3] + 0.05 * losses[4]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            last_loss = float(loss.detach().cpu())
    return model.eval(), {"optimizer_step_calls": updates,
                          "training_sample_presentations": EPOCHS * n,
                          "last_loss": last_loss}


def _f1(target: np.ndarray, predicted: np.ndarray, values: tuple[int, ...]) -> float:
    scores = []
    for value in values:
        tp = int(((target == value) & (predicted == value)).sum())
        fp = int(((target != value) & (predicted == value)).sum())
        fn = int(((target == value) & (predicted != value)).sum())
        if 2 * tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluate(model: PublicNextFramePredictor, rows: dict[str, np.ndarray]) -> dict:
    n = len(rows["features"])
    pieces = []
    for start in range(0, n, BATCH_SIZE):
        logits = model(torch.from_numpy(rows["features"][start:start + BATCH_SIZE]))
        pieces.append(logits[0].argmax(-1).cpu().numpy())
    predicted = np.concatenate(pieces)
    actual = rows["next_obj"]
    identity = rows["prior_obj"]
    eligible = np.isin(rows["action"], [3, 4, 5])
    changed = eligible & np.any(actual != identity, axis=1)
    interesting = tuple(v for k, v in OBJECT_TO_IDX.items() if k in
                        ("door", "key", "box", "ball", "wall"))
    return {
        "monitor_transitions": n,
        "native_interaction_action_steps": int(eligible.sum()),
        "public_changed_interaction_steps": int(changed.sum()),
        "next_object_cell_accuracy": float(np.mean(actual == predicted)),
        "persistence_next_object_cell_accuracy": float(np.mean(actual == identity)),
        "interaction_macro_f1": _f1(actual[eligible], predicted[eligible], interesting),
        "persistence_interaction_macro_f1": _f1(actual[eligible], identity[eligible], interesting),
        "changed_interaction_macro_f1": _f1(actual[changed], predicted[changed], interesting),
        "persistence_changed_interaction_macro_f1": _f1(actual[changed], identity[changed], interesting),
    }


def save_numeric(model: PublicNextFramePredictor, path: Path) -> str:
    if path.exists():
        raise FileExistsError("Refusing to overwrite M23 learned numeric state")
    arrays = {f"tensor_{i:03d}": t.detach().cpu().numpy()
              for i, t in enumerate(model.state_dict().values())}
    if not arrays or any(x.dtype.kind not in "fiub" for x in arrays.values()):
        raise ValueError("M23 nonnumeric state is forbidden")
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(input_root: str, output: str) -> dict:
    validate_split_contract()
    source_sha = admitted_source_commit()
    root = Path(output)
    if str(root) != f"{OUT}/summary.json" or root.exists():
        raise ValueError("M23 output must be new in registered ignored location")
    root.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    original_report, training_public = load_source_artifact(Path(input_root))
    train = combine([encode_public_rows(training_public[task]) for task in SOURCE_ENV_IDS])
    monitor, physical_monitor = source_monitor_data()
    report = {
        "stage": "m23_fresh_source_world_model_predictor",
        "source_commit": source_sha,
        "cached_training_source_commit": TRAIN_SOURCE_COMMIT,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "cached_training_native_actions": original_report["actual_native_actions"],
        "actual_native_actions_this_run": sum(x["actual_native_actions"]
                                               for x in physical_monitor.values()),
        "new_source_monitor_physical_effects": physical_monitor,
        "target_env_actions": 0, "target_final_episodes": 0,
        "fresh_model_fits": len(MODEL_SEEDS), "optimizer_step_calls": 0,
        "model_seeds": list(MODEL_SEEDS), "epochs_per_fit": EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": LR,
        "library_versions": {k: version(k) for k in ("torch", "numpy", "minigrid", "gymnasium")},
        "seed_results": {},
    }
    for seed in MODEL_SEEDS:
        model, trained = fit_one(train, seed)
        metrics = evaluate(model, monitor)
        path = root.parent / f"source-public-world-model-seed{seed}.npz"
        digest = save_numeric(model, path)
        report["optimizer_step_calls"] += trained["optimizer_step_calls"]
        report["seed_results"][str(seed)] = {**trained, **metrics,
                                              "checkpoint_sha256": digest}
        print(json.dumps({"model_seed": seed, **trained, **metrics}), flush=True)
    baseline = [v["persistence_interaction_macro_f1"]
                for v in report["seed_results"].values()]
    gains = [v["interaction_macro_f1"] - b for v, b in
             zip(report["seed_results"].values(), baseline, strict=True)]
    report["registered_source_prediction_gate_pass"] = bool(
        sum(g > 0.05 for g in gains) >= 6
        and float(np.mean(gains)) >= 0.05
        and monitor["features"].shape[0] > 0
    )
    report["mean_interaction_macro_f1_gain"] = float(np.mean(gains))
    report["elapsed_seconds"] = time.perf_counter() - started
    root.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fresh_model_fits": report["fresh_model_fits"],
                      "optimizer_step_calls": report["optimizer_step_calls"],
                      "actual_native_actions_this_run": report["actual_native_actions_this_run"],
                      "cached_training_native_actions": report["cached_training_native_actions"],
                      "mean_interaction_macro_f1_gain": report["mean_interaction_macro_f1_gain"],
                      "registered_source_prediction_gate_pass":
                      report["registered_source_prediction_gate_pass"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=f"{OUT}/input")
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("M23 fresh world-model fit requires --execute")
    execute(args.input_root, args.output)


if __name__ == "__main__":
    main()
