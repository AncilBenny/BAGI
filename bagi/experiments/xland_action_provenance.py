"""REUSE-19: test DAgger teacher-label versus EXECUTED-action provenance.

Use upstream SimpleDAggerTrainer without alteration; record labels offered by
the privileged expert and the action truly sent to upstream XLand. Only the
latter can be used as a past-action feature at learner deployment. Training
skills only. No held-out causal-chain oracle access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_existing_bc import TRAIN_TASKS, visible_features
from bagi.experiments.xland_existing_dagger import (
    TrainingXLandGym,
    make_privileged_expert,
)

PAD_ACTION = 6
WINDOWS = (0, 1, 4)


class LoggedTrainingGym(TrainingXLandGym):
    """Expose real executed steps without changing upstream XLand mechanics."""

    def __init__(self, *, seed_base, cap):
        super().__init__(seed_base=seed_base, cap=cap)
        self.pending_teacher_label = None
        self.pending_round = None
        self.episode_step = 0
        self.records = []

    def reset(self, *, seed=None, options=None):
        if self.pending_teacher_label is not None:
            raise AssertionError("Unused expert label survived episode reset")
        obs, info = super().reset(seed=seed, options=options)
        self.episode_step = 0
        return obs, info

    def attach_label(self, label, round_num):
        if self.pending_teacher_label is not None:
            raise AssertionError("Double expert label without simulator step")
        if self.task not in TRAIN_TASKS or int(label) not in range(6):
            raise ValueError("Only native actions on original train skills")
        self.pending_teacher_label = int(label)
        self.pending_round = int(round_num)

    def step(self, action):
        if self.pending_teacher_label is None:
            raise AssertionError("Real simulator step missing expert label")
        action = int(np.asarray(action).item())
        record = {
            "round": self.pending_round,
            "episode": self.episode_index - 1,
            "episode_step": self.episode_step,
            "task": self.task,
            "obs": visible_features(self.timestep.observation).copy(),
            "expert_label": self.pending_teacher_label,
            "executed_action": action,
        }
        response = super().step(action)
        record["done"] = bool(response[2] or response[3])
        record["rewarded"] = bool(response[1] > 0)
        self.records.append(record)
        self.episode_step += 1
        self.pending_teacher_label = None
        self.pending_round = None
        return response


def make_logged_expert(original, room, read_round):
    """Delegate all expert choices to unchanged REUSE-14 teacher."""
    from stable_baselines3.common.policies import BasePolicy

    class LoggedExpert(BasePolicy):
        def __init__(self):
            super().__init__(
                observation_space=original.observation_space,
                action_space=original.action_space,
            )
            self.delegate = original

        def _predict(self, observation, deterministic=False):
            output = self.delegate._predict(
                observation, deterministic=deterministic
            )
            label = int(output.detach().cpu().numpy().reshape(-1)[0])
            room.attach_label(label, read_round())
            return output

    return LoggedExpert()


def add_preceding_histories(records):
    """Reset at episode boundary; current oracle label NEVER enters its history."""
    last_episode = None
    executed, offered = [], []
    output = []
    for row in records:
        episode = (row["round"], row["episode"])
        if episode != last_episode:
            executed, offered = [], []
            last_episode = episode
        if row["episode_step"] != len(executed):
            raise AssertionError("Non-contiguous episode step provenance")
        item = dict(row)
        item["prior_executed"] = tuple(
            ([PAD_ACTION] * 4 + executed)[-4:]
        )
        item["prior_expert"] = tuple(
            ([PAD_ACTION] * 4 + offered)[-4:]
        )
        output.append(item)
        executed.append(int(row["executed_action"]))
        offered.append(int(row["expert_label"]))
        if bool(row["done"]):
            last_episode = None
    return output


def signature_audit(rows, *, stream, window):
    """In-sample majority lookup diagnostic, NOT a train/generalization ceiling."""
    if stream not in ("executed", "expert") or window not in WINDOWS:
        raise ValueError("Unsupported history stream or window")
    grouped = defaultdict(Counter)
    for row in rows:
        visible = np.ascontiguousarray(
            np.asarray(row["obs"], dtype=np.float32)
        ).tobytes()
        context = row["prior_" + stream][-window:] if window else ()
        grouped[(visible, context)][int(row["expert_label"])] += 1
    conflicting = [
        counts for counts in grouped.values() if len(counts) > 1
    ]
    source_n = len(rows)
    maximum = sum(max(counts.values()) for counts in grouped.values())
    return {
        "history_stream": stream,
        "preceding_window": window,
        "source_rows": source_n,
        "distinct_signatures": len(grouped),
        "repeated_groups": sum(sum(c.values()) > 1 for c in grouped.values()),
        "contradictory_groups": len(conflicting),
        "contradictory_rows": sum(sum(c.values()) for c in conflicting),
        "majority_lookup_in_sample_correct": maximum,
        "majority_lookup_in_sample_accuracy": maximum / source_n
        if source_n else None,
        "minority_label_errors": source_n - maximum,
    }


def audit(records, *, expected_rounds=(0, 1)):
    rows = add_preceding_histories(records)
    summary = {}
    for round_num in expected_rounds:
        selected = [r for r in rows if r["round"] == round_num]
        if not selected:
            raise AssertionError("Expected nonempty DAgger round")
        n = len(selected)
        mismatches = sum(r["expert_label"] != r["executed_action"]
                         for r in selected)
        if round_num == 0 and mismatches:
            raise AssertionError("Beta=1 teacher-executed step was not identical")
        summary[str(round_num)] = {
            "steps": n,
            "expert_action_counts": dict(Counter(
                r["expert_label"] for r in selected
            )),
            "executed_action_counts": dict(Counter(
                r["executed_action"] for r in selected
            )),
            "expert_execution_mismatches": mismatches,
            "expert_execution_mismatch_rate": mismatches / n,
            "rewarded_goal_events": sum(r["rewarded"] for r in selected),
            "actual_action_history_lookups": {
                str(k): signature_audit(
                    selected, stream="executed", window=k
                ) for k in WINDOWS
            },
            "wrong_counterfactual_expert_history_lookup": signature_audit(
                selected, stream="expert", window=4
            ),
        }
    if len(rows) != sum(s["steps"] for s in summary.values()):
        raise AssertionError("Extra unaccounted simulator interactions")
    return rows, summary


def numeric_audit_arrays(rows):
    """Persist only numeric observable fields/label streams; no object pickle."""
    return {
        "obs": np.stack([r["obs"] for r in rows]).astype(np.float32),
        "expert_label": np.asarray(
            [r["expert_label"] for r in rows], dtype=np.int64
        ),
        "executed_action": np.asarray(
            [r["executed_action"] for r in rows], dtype=np.int64
        ),
        "prior_executed_actions": np.asarray(
            [r["prior_executed"] for r in rows], dtype=np.int64
        ),
        "round": np.asarray([r["round"] for r in rows], dtype=np.int64),
        "episode": np.asarray([r["episode"] for r in rows], dtype=np.int64),
        "step_in_episode": np.asarray(
            [r["episode_step"] for r in rows], dtype=np.int64
        ),
        "done": np.asarray([r["done"] for r in rows], dtype=np.bool_),
    }


def run(config_path):
    import torch
    from imitation.algorithms.bc import BC
    from imitation.algorithms.dagger import LinearBetaSchedule, SimpleDAggerTrainer
    from imitation.data import rollout
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["training_tasks"] != list(TRAIN_TASKS)
            or cfg["model_seeds"] != [17, 29]
            or cfg["training_seed_base"] != 7500
            or cfg["rollout_min_steps"] != 256
            or cfg["round_min_steps"] != 128
            or cfg["round_min_episodes"] != 3
            or cfg["batch_size"] != 32
            or cfg["bc_epochs"] != 4
            or cfg["cap"] != 40
            or cfg["max_online_steps_per_model"] != 400):
        raise ValueError("Frozen REUSE-19 DAgger collection contract changed")
    torch.set_num_threads(1)
    results = []
    folder = Path(cfg["output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    for seed in cfg["model_seeds"]:
        room = LoggedTrainingGym(seed_base=7500, cap=40)
        venv = DummyVecEnv([lambda env=room: env])
        original_expert = make_privileged_expert(room)
        round_box = [0]
        expert = make_logged_expert(original_expert, room, lambda: round_box[0])
        torch.manual_seed(seed)
        bc = BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            rng=np.random.default_rng(seed),
            batch_size=32,
            device="cpu",
        )
        with tempfile.TemporaryDirectory(prefix="bagi_action_audit_") as temp:
            trainer = SimpleDAggerTrainer(
                venv=venv, scratch_dir=temp, expert_policy=expert,
                bc_trainer=bc, rng=np.random.default_rng(seed),
                beta_schedule=LinearBetaSchedule(rampdown_rounds=1),
            )
            # Read current upstream round each time teacher selects its label.
            round_box[0] = 0
            original_predict = expert._predict

            def with_round(observation, deterministic=False):
                round_box[0] = int(trainer.round_num)
                return original_predict(
                    observation, deterministic=deterministic
                )

            expert._predict = with_round
            trainer.train(
                total_timesteps=256,
                rollout_round_min_episodes=3,
                rollout_round_min_timesteps=128,
                bc_train_kwargs={
                    "n_epochs": 4, "progress_bar": False,
                    "log_rollouts_venv": None,
                    "log_rollouts_n_episodes": 0,
                },
            )
            if trainer.round_num != 2:
                raise AssertionError("Original upstream DAgger round count changed")
            if room.pending_teacher_label is not None:
                raise AssertionError("Pending unlabeled simulator step")
            if len(room.records) != room.real_steps:
                raise AssertionError("Actual-step instrumentation not exhaustive")
            if original_expert.teacher_queries != room.real_steps:
                raise AssertionError("Expert and actual simulator steps differ")
            if room.real_steps > 400:
                raise AssertionError("Training interaction cap exceeded")
            aggregate = rollout.flatten_trajectories(trainer._all_demos)
            teacher_multiset = Counter(
                (np.ascontiguousarray(o).tobytes(), int(a))
                for o, a in zip(aggregate.obs, aggregate.acts)
            )
            executed_multiset = Counter(
                (np.ascontiguousarray(r["obs"]).tobytes(),
                 int(r["expert_label"]))
                for r in room.records
            )
            if teacher_multiset != executed_multiset:
                raise AssertionError(
                    "Upstream DAgger stored expert-label stream differs "
                    "from the labeled simulator-visit records"
                )
        rows, per_round = audit(room.records)
        numeric = numeric_audit_arrays(rows)
        checksum = hashlib.sha256()
        for name, arr in sorted(numeric.items()):
            checksum.update(name.encode())
            checksum.update(np.ascontiguousarray(arr).tobytes())
        np.savez_compressed(folder / f"action_provenance_seed{seed}.npz", **numeric)
        results.append({
            "model_seed": seed,
            "real_online_training_steps": room.real_steps,
            "privileged_train_only_oracle_queries": original_expert.teacher_queries,
            "episodes_by_train_task": dict(room.episodes_by_task),
            "stored_teacher_labels_match_logged_visits": True,
            "numeric_record_sha256": checksum.hexdigest(),
            "actual_preceding_actions_reset_at_episode_boundaries": True,
            "rounds": per_round,
        })
        print(f"seed={seed}: online={room.real_steps} "
              f"round1 mismatch={per_round['1']['expert_execution_mismatches']}/"
              f"{per_round['1']['steps']} "
              f"real-history contradictions={per_round['1']['actual_action_history_lookups']['4']['contradictory_groups']}")
        venv.close()
    summary = {
        "question": "Do expert labels diverge from simulator-executed actions in DAgger, and does valid learner history change teacher-label aliasing?",
        "method": "Unchanged upstream imitation.SimpleDAggerTrainer; minimal action provenance instrumentation",
        "packages": {"imitation": version("imitation")},
        "training_tasks_only": list(TRAIN_TASKS),
        "heldout_teacher_labels": 0,
        "new_heldout_simulator_steps": 0,
        "privileged_inputs_supplied_to_learner": 0,
        "beta_round0": 1.0,
        "beta_round1": 0.0,
        "history_action_pad_value": PAD_ACTION,
        "windows": list(WINDOWS),
        "model_results": results,
        "total_actual_training_steps": sum(r["real_online_training_steps"] for r in results),
        "total_expert_queries": sum(r["privileged_train_only_oracle_queries"] for r in results),
        "limitations": [
            "DAgger acquisition is repeated only to instrument previously unavailable action execution streams; this is not a new independent skill/generalization test.",
            "The teacher's output is the label for CURRENT state, not an executed action during beta=0.",
            "History signatures contain only preceding ACTUAL actions reset on episode boundaries; current teacher label never enters the history.",
            "In-sample majority memorization ceilings can increase merely because contexts become more unique; not evidence a learned recurrent policy would improve.",
            "Comparing executed vs teacher action histories is a representation-validity diagnostic, not a fitted closed-loop learner.",
            "No heldout causal chain, privilege at runtime learner features, external pickle load or new model architecture.",
        ],
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Saved provenance and action-history audit")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_action_provenance.yaml")
    run(p.parse_args().config)
