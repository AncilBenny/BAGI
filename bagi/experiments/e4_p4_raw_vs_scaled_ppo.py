"""G2-E4-P4: registered raw vs /10 PUBLIC categorical-input PPO experiment.

Only upstream SB3 PPO and official Farama MiniGrid are used. Frozen equal
per-model training budgets, matched seeds and a single input scale contrast.
Evaluator-only physical hashes never enter actor inputs, learning, or actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

TASK = "MiniGrid-Empty-Random-6x6-v0"
MODEL_SEEDS = (17, 29)
SCALES = ("raw", "scaled")
FEEDBACK = ("live", "stale")
TEST_SEEDS = tuple(range(29000, 29016))
TRAIN_BASE = 2_000_000
TRAIN_STRIDE = 100_000
TRAIN_PER_MODEL = 16_384
ROLLOUT_STEPS = 128
CAP = 144
N_ACTIONS = 7
FEATURES = 147
NORMALIZER = 10.0
EVAL_RNG_BASE = 29_000_000
POLICY_KWARGS = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
TRAIN_CEILING = len(MODEL_SEEDS) * len(SCALES) * TRAIN_PER_MODEL  # 65,536
TEST_CEILING = (
    len(MODEL_SEEDS) * len(SCALES) * len(FEEDBACK) * len(TEST_SEEDS) * CAP
)  # 18,432
ALL_CEILING = TRAIN_CEILING + TEST_CEILING  # 83,968


def verify_registration() -> None:
    """Fail closed on any retrospectively modified task, seeds or resources."""
    if (
        TASK != "MiniGrid-Empty-Random-6x6-v0"
        or MODEL_SEEDS != (17, 29)
        or SCALES != ("raw", "scaled")
        or FEEDBACK != ("live", "stale")
        or TEST_SEEDS != tuple(range(29000, 29016))
        or TRAIN_BASE != 2_000_000
        or TRAIN_STRIDE != 100_000
        or TRAIN_PER_MODEL != 16_384
        or ROLLOUT_STEPS != 128
        or CAP != 144
        or N_ACTIONS != 7
        or FEATURES != 147
        or NORMALIZER != 10.0
        or EVAL_RNG_BASE != 29_000_000
        or POLICY_KWARGS != {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
        or TRAIN_CEILING != 65_536
        or TEST_CEILING != 18_432
        or ALL_CEILING != 83_968
    ):
        raise ValueError("G2-E4-P4 frozen preregistration changed")


def native_public(raw: Any) -> np.ndarray:
    """Validate flattened categorical MiniGrid view without oracle information."""
    arr = np.asarray(raw)
    if (
        arr.shape != (FEATURES,)
        or arr.dtype.kind not in "biu"
        or not np.isfinite(arr).all()
        or np.any(arr < 0)
        or np.any(arr > NORMALIZER)
    ):
        raise ValueError("Expected 147 finite PUBLIC categorical IDs 0..10")
    return np.ascontiguousarray(arr, dtype=np.float32)


def actor_public(raw: Any, scale: str) -> np.ndarray:
    """Only arm difference: public numeric magnitude, never extra information."""
    if scale not in SCALES:
        raise ValueError("Unregistered input scale")
    values = native_public(raw)
    if scale == "scaled":
        values /= NORMALIZER
    return values


def _hash_layout(env: Any) -> str:
    """Evaluator-only reset diversity; never accessed by a policy call."""
    world = env.unwrapped
    arr = np.ascontiguousarray(world.grid.encode())
    pos = tuple(int(x) for x in world.agent_pos)
    direction = int(world.agent_dir)
    if arr.ndim != 3 or arr.dtype.kind not in "biu" or len(pos) != 2:
        raise ValueError("Unexpected evaluator-only native grid")
    if direction not in range(4):
        raise ValueError("Unexpected native orientation")
    h = hashlib.sha256()
    h.update(json.dumps([arr.shape, str(arr.dtype), pos, direction]).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _public_hash(raw: Any) -> str:
    arr = native_public(raw)
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()


def sampling_seed(model_seed: int, reset_seed: int) -> int:
    if model_seed not in MODEL_SEEDS or reset_seed not in TEST_SEEDS:
        raise ValueError("Unregistered evaluation sampling seed")
    return EVAL_RNG_BASE + MODEL_SEEDS.index(model_seed) * 100_000 + reset_seed - TEST_SEEDS[0]


def make_world() -> Any:
    """Official Gymnasium 147-code public view, no mission/full-grid input."""
    import gymnasium as gym
    import minigrid  # noqa: F401 - official benchmark registration
    from gymnasium.wrappers import FlattenObservation
    from minigrid.wrappers import ImgObsWrapper

    world = FlattenObservation(ImgObsWrapper(gym.make(TASK)))
    if world.observation_space.shape != (FEATURES,) or int(world.action_space.n) != N_ACTIONS:
        world.close()
        raise ValueError("Unexpected standard MiniGrid public shape or native actions")
    return world


def make_train_env(scale: str, model_seed: int, ledger: dict[str, int]) -> Any:
    """Native counting occurs at the only genuine upstream env.step boundary."""
    import gymnasium as gym
    from gymnasium import spaces

    if model_seed not in MODEL_SEEDS or scale not in SCALES:
        raise ValueError("Unregistered train model condition")
    reset_base = TRAIN_BASE + MODEL_SEEDS.index(model_seed) * TRAIN_STRIDE

    class NativeTrainRecorder(gym.Wrapper):
        def __init__(self, env: Any) -> None:
            super().__init__(env)
            upper = 1.0 if scale == "scaled" else NORMALIZER
            self.observation_space = spaces.Box(
                low=0.0, high=upper, shape=(FEATURES,), dtype=np.float32
            )
            self.real_actions = 0
            self.native_resets = 0
            self.native_goals = 0
            self.completed_episodes = 0
            self.train_reset_seeds: list[int] = []
            self.train_reset_layouts: list[str] = []
            self.rows: dict[str, list] = {
                "public_codes": [], "actor_input": [], "actions": [],
                "native_rewards": [], "terminated": [], "truncated": [],
                "episode_index": [],
            }
            self._raw: np.ndarray | None = None

        def reset(self, **_kwargs: Any) -> tuple[np.ndarray, dict]:
            # Ignore PPO's model seed; always use matched, disjoint native stream.
            native_seed = reset_base + self.native_resets
            if self.native_resets >= TRAIN_STRIDE:
                raise ValueError("Registered TRAIN reset stream exhausted")
            raw, _private = self.env.reset(seed=native_seed)
            self.native_resets += 1
            self.train_reset_seeds.append(native_seed)
            self.train_reset_layouts.append(_hash_layout(self.env))
            self._raw = native_public(raw)
            return actor_public(raw, scale), {}

        def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
            if self._raw is None:
                raise ValueError("TRAIN step before native reset")
            if self.real_actions >= TRAIN_PER_MODEL or ledger["train"] >= TRAIN_CEILING:
                raise ValueError("Native TRAIN action budget exceeded")
            if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
                raise TypeError("Expected a native discrete action")
            a = int(action)
            if not self.action_space.contains(a):
                raise ValueError("Illegal native TRAIN action")
            prev_raw = self._raw.copy()
            next_obs, reward, terminated, truncated, _info = self.env.step(a)
            next_raw = native_public(next_obs)
            raw_reward = float(reward)
            if not np.isfinite(raw_reward):
                raise ValueError("Nonfinite native TRAIN reward")
            self.rows["public_codes"].append(prev_raw)
            self.rows["actor_input"].append(actor_public(prev_raw.astype(np.uint8), scale))
            self.rows["actions"].append(a)
            self.rows["native_rewards"].append(raw_reward)
            self.rows["terminated"].append(int(terminated))
            self.rows["truncated"].append(int(truncated))
            self.rows["episode_index"].append(self.native_resets - 1)
            self._raw = next_raw
            self.real_actions += 1
            ledger["train"] += 1
            self.native_goals += int(bool(terminated) and raw_reward > 0)
            self.completed_episodes += int(bool(terminated) or bool(truncated))
            return (
                actor_public(next_raw.astype(np.uint8), scale), raw_reward,
                bool(terminated), bool(truncated), {},
            )

    return NativeTrainRecorder(make_world())


def test_episode(
    world: Any, policy: Any, scale: str, feedback: str, model_seed: int,
    reset_seed: int, ledger: dict[str, int],
) -> tuple[dict, dict[str, np.ndarray]]:
    """One native stochastic rollout under original or frozen-initial PUBLIC view."""
    import torch

    if scale not in SCALES or feedback not in FEEDBACK:
        raise ValueError("Unregistered evaluation arm")
    if model_seed not in MODEL_SEEDS or reset_seed not in TEST_SEEDS:
        raise ValueError("Unregistered TEST model/reset seed")
    world.action_space.seed(reset_seed)
    raw, _private = world.reset(seed=reset_seed)
    initial_raw = native_public(raw).copy()
    layout = _hash_layout(world)
    first_public = _public_hash(raw)
    rng_seed = sampling_seed(model_seed, reset_seed)
    torch.manual_seed(rng_seed)
    rows: dict[str, list] = {
        "public_codes": [], "actor_input": [], "actions": [],
        "native_rewards": [], "terminated": [], "truncated": [],
    }
    terminated = truncated = False
    for _ in range(CAP):
        actual = native_public(raw)
        supplied = actual if feedback == "live" else initial_raw
        policy_input = actor_public(supplied.astype(np.uint8), scale)
        action, _state = policy.predict(policy_input, deterministic=False)
        a = int(np.asarray(action).item())
        if not world.action_space.contains(a):
            raise ValueError("Invalid action from public stochastic actor")
        if ledger["test"] >= TEST_CEILING:
            raise ValueError("Native TEST action budget exceeded")
        next_raw, reward, terminated, truncated, _info = world.step(a)
        ledger["test"] += 1
        rew = float(reward)
        if not np.isfinite(rew):
            raise ValueError("Nonfinite native TEST reward")
        rows["public_codes"].append(actual.copy())
        rows["actor_input"].append(policy_input.copy())
        rows["actions"].append(a)
        rows["native_rewards"].append(rew)
        rows["terminated"].append(int(terminated))
        rows["truncated"].append(int(truncated))
        raw = next_raw
        if terminated or truncated:
            break
    if not (terminated or truncated):
        raise ValueError("Upstream native TEST episode exceeded 144 actions")
    native_reward = float(sum(rows["native_rewards"]))
    if native_reward > 0 and not terminated:
        raise ValueError("Positive native reward without termination")
    case = {
        "scale": scale, "feedback": feedback, "model_seed": model_seed,
        "reset_seed": reset_seed, "rng_seed": rng_seed,
        "layout_sha256": layout, "initial_public_sha256": first_public,
        "actions": len(rows["actions"]), "native_reward": round(native_reward, 8),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "native_goal": int(bool(terminated) and native_reward > 0),
        "action_counts": np.bincount(rows["actions"], minlength=N_ACTIONS).tolist(),
    }
    return case, {k: np.asarray(v) for k, v in rows.items()}


def _pack(
    path: Path, episodes: list[dict[str, np.ndarray]],
    weights: dict[str, np.ndarray] | None = None,
    initial_weights: dict[str, np.ndarray] | None = None,
) -> str:
    """Safe, per-action numeric evidence; no pickle or serialized model object."""
    if not episodes:
        raise ValueError("Cannot archive empty evidence")
    arrays = {
        "episode_offsets": np.asarray(
            [0] + list(np.cumsum([len(e["actions"]) for e in episodes])),
            dtype=np.int64,
        ),
    }
    for key in episodes[0]:
        arrays[key] = np.concatenate([e[key] for e in episodes], axis=0)
    for key, value in (weights or {}).items():
        archive_key = "policy__" + key.replace(".", "__")
        if archive_key in arrays:
            raise ValueError("Policy tensor archive name collision")
        arrays[archive_key] = np.asarray(value)
    for key, value in (initial_weights or {}).items():
        archive_key = "initial_policy__" + key.replace(".", "__")
        if archive_key in arrays:
            raise ValueError("Initial policy tensor archive name collision")
        arrays[archive_key] = np.asarray(value)
    if any(a.dtype.kind not in "biuf" or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Unsafe or nonfinite evidence dtype/value")
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights(policy: Any) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy()
            for key, value in policy.state_dict().items()}


def _weights_digest(weights: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        digest.update(key.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def first_layer_saturation(inputs: np.ndarray, weights: dict[str, np.ndarray]) -> dict:
    """TRAIN-only descriptive Tanh diagnosis; never used for policy decisions."""
    x = np.asarray(inputs, dtype=np.float64)
    w = weights["mlp_extractor.policy_net.0.weight"].astype(np.float64)
    b = weights["mlp_extractor.policy_net.0.bias"].astype(np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURES or not np.isfinite(x).all():
        raise ValueError("Invalid public TRAIN observation sensitivity inputs")
    activation = np.tanh(x @ w.T + b)
    return {
        "train_views": len(x),
        "fraction_abs_tanh_above_0_95": round(float(np.mean(np.abs(activation) > 0.95)), 6),
        "mean_tanh_derivative": round(float(np.mean(1 - activation * activation)), 6),
    }


def _train_seed_pair_check(results: list[dict]) -> None:
    for model_seed in MODEL_SEEDS:
        matches = [x for x in results if x["model_seed"] == model_seed]
        if len(matches) != len(SCALES) or len({x["initial_state_sha256"] for x in matches}) != 1:
            raise ValueError("Input comparison changed initial network parameters")
        first_seeds = [r["train"]["reset_seeds"][0] for r in matches]
        if len({r["train"]["first_public_sha256"] for r in matches}) != 1:
            raise ValueError("Raw and scaled training starts used different public views")
        if first_seeds != [TRAIN_BASE + MODEL_SEEDS.index(model_seed) * TRAIN_STRIDE] * 2:
            raise ValueError("Scale arms did not use registered matched native starts")


def score(results: list[dict]) -> dict:
    """Apply the single-variable, per-seed preregistered stop gate."""
    if len(results) != len(MODEL_SEEDS) * len(SCALES):
        raise ValueError("Missing scale/model conditions")
    out = {}
    for model_seed in MODEL_SEEDS:
        by_scale = {r["scale"]: r for r in results if r["model_seed"] == model_seed}
        if set(by_scale) != set(SCALES):
            raise ValueError("Missing model-seed scale comparison")
        r, s = by_scale["raw"], by_scale["scaled"]
        for x in (r, s):
            if x["train"]["real_actions"] != TRAIN_PER_MODEL or x["train"]["native_goals"] < 0:
                raise ValueError("Unaccounted native TRAIN interactions")
            if set(x["eval"]) != set(FEEDBACK):
                raise ValueError("Missing evaluation feedback arm")
            for arm in FEEDBACK:
                cases = x["eval"][arm]
                if len(cases) != len(TEST_SEEDS):
                    raise ValueError("Missing registered TEST episodes")
                if [c["reset_seed"] for c in cases] != list(TEST_SEEDS):
                    raise ValueError("Unregistered or reordered TEST starts")
                if len({c["layout_sha256"] for c in cases}) < 12:
                    raise ValueError("Insufficient distinct physical TEST starts")
        for index, seed in enumerate(TEST_SEEDS):
            all_cases = [x["eval"][arm][index] for x in (r, s) for arm in FEEDBACK]
            if len({c["layout_sha256"] for c in all_cases}) != 1:
                raise ValueError("Scale/feedback arms did not share native layout")
            if len({c["initial_public_sha256"] for c in all_cases}) != 1:
                raise ValueError("Scale/feedback arms did not share initial public view")
            if len({c["rng_seed"] for c in all_cases}) != 1:
                raise ValueError("Scale/feedback arms did not share sampling stream")
            if any(c["rng_seed"] != sampling_seed(model_seed, seed) for c in all_cases):
                raise ValueError("Sampling seed deviated from frozen registration")
            if any(c["native_goal"] != int(c["native_reward"] > 0 and c["terminated"])
                   or not 1 <= c["actions"] <= CAP
                   or sum(c["action_counts"]) != c["actions"] for c in all_cases):
                raise ValueError("Inconsistent real action and native goal accounting")
        goals = {
            scale: {feedback: sum(x["native_goal"] for x in by_scale[scale]["eval"][feedback])
                    for feedback in FEEDBACK}
            for scale in SCALES
        }
        feedback_gains = {scale: goals[scale]["live"] - goals[scale]["stale"]
                          for scale in SCALES}
        scaled_live_wins = sum(
            a["native_goal"] > b["native_goal"]
            for a, b in zip(s["eval"]["live"], s["eval"]["stale"], strict=True)
        )
        scaled_live_losses = sum(
            a["native_goal"] < b["native_goal"]
            for a, b in zip(s["eval"]["live"], s["eval"]["stale"], strict=True)
        )
        out[str(model_seed)] = {
            "native_goals": goals,
            "live_minus_stale": feedback_gains,
            "scaled_live_wins": scaled_live_wins,
            "scaled_live_losses": scaled_live_losses,
            "distinct_layouts": len({x["layout_sha256"] for x in s["eval"]["live"]}),
            "improved_feedback_signal": bool(
                r["train"]["native_goals"] > 0 and s["train"]["native_goals"] > 0
                and goals["scaled"]["live"] >= 8
                and goals["scaled"]["live"] >= goals["raw"]["live"]
                and feedback_gains["scaled"] >= 3
                and feedback_gains["scaled"] > feedback_gains["raw"]
                and scaled_live_losses <= 1
            ),
        }
    return {
        "seed_results": out,
        "strong_feedback_improvement_both_seeds": all(
            out[str(seed)]["improved_feedback_signal"] for seed in MODEL_SEEDS
        ),
    }


def run(output_dir: Path) -> dict:
    """One frozen TRAIN and TEST run, exact real-step ledger, no teacher or probes."""
    import torch
    from stable_baselines3 import PPO

    verify_registration()
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = {"train": 0, "test": 0}
    summary: dict[str, Any] = {
        "study": "G2-E4-P4", "status": "running", "type": "single-variable input-scale PPO",
        "task": TASK, "scales": list(SCALES), "feedback_arms": list(FEEDBACK),
        "model_seeds": list(MODEL_SEEDS), "test_seeds": list(TEST_SEEDS),
        "train_steps_per_model": TRAIN_PER_MODEL,
        "max_native_train_actions": TRAIN_CEILING, "max_native_test_actions": TEST_CEILING,
        "max_all_native_actions": ALL_CEILING, "native_action_ledger": ledger,
        "teacher_queries": 0, "ppo_rollouts": 0, "results": [],
        "numeric_artifacts": [], "promotion": False,
        "versions": {k: version(k) for k in ("gymnasium", "minigrid", "stable-baselines3")},
        "limitations": [
            "One simple variable-start navigation task, not transfer or general intelligence.",
            "Only raw vs /10 public categorical ID magnitude changes; not one-hot semantics.",
            "Old P1-P3 heldout seeds were never used by this registered experiment.",
            "Small paired evaluation; a pass requires independent confirmation.",
        ],
    }
    try:
        # Model seed outermost; raw then scaled use matched initial parameters.
        for model_seed in MODEL_SEEDS:
            for scale in SCALES:
                train_env = make_train_env(scale, model_seed, ledger)
                try:
                    kwargs = {
                        "seed": model_seed, "device": "cpu", "n_steps": ROLLOUT_STEPS,
                        "batch_size": 64, "n_epochs": 2, "gamma": 0.99,
                        "learning_rate": 3e-4, "ent_coef": 0.01,
                        "policy_kwargs": POLICY_KWARGS, "verbose": 0,
                    }
                    model = PPO("MlpPolicy", train_env, **kwargs)
                    initial_weights = _weights(model.policy)
                    initial_sha = _weights_digest(initial_weights)
                    model.learn(total_timesteps=TRAIN_PER_MODEL)
                    if int(model.num_timesteps) != TRAIN_PER_MODEL:
                        raise ValueError("Incorrect upstream PPO real timesteps")
                    if train_env.real_actions != TRAIN_PER_MODEL:
                        raise ValueError("Incorrect actual native TRAIN action count")
                    final_weights = _weights(model.policy)
                    train_evidence = {
                        "public_codes": np.stack(train_env.rows["public_codes"]),
                        "actor_input": np.stack(train_env.rows["actor_input"]),
                        "actions": np.asarray(train_env.rows["actions"], dtype=np.int16),
                        "native_rewards": np.asarray(
                            train_env.rows["native_rewards"], dtype=np.float64
                        ),
                        "terminated": np.asarray(train_env.rows["terminated"], dtype=np.int8),
                        "truncated": np.asarray(train_env.rows["truncated"], dtype=np.int8),
                        "episode_index": np.asarray(
                            train_env.rows["episode_index"], dtype=np.int32
                        ),
                    }
                    training_file = output_dir / f"train_seed{model_seed}_{scale}.npz"
                    train_sha = _pack(
                        training_file, [train_evidence], final_weights, initial_weights
                    )
                    summary["numeric_artifacts"].append({
                        "name": training_file.name, "sha256": train_sha,
                        "kind": "all native TRAIN steps and final policy numeric weights",
                    })
                    details = {
                        "model_seed": model_seed, "scale": scale,
                        "initial_state_sha256": initial_sha,
                        "final_state_sha256": _weights_digest(final_weights),
                        "train": {
                            "real_actions": train_env.real_actions,
                            "native_goals": train_env.native_goals,
                            "native_completed_episodes": train_env.completed_episodes,
                            "native_resets": train_env.native_resets,
                            "unique_train_reset_layouts": len(set(train_env.train_reset_layouts)),
                            "reset_seeds": train_env.train_reset_seeds,
                            "first_public_sha256": _public_hash(
                                train_evidence["public_codes"][0].astype(np.uint8)
                            ),
                            "initial_saturation_train_1024": first_layer_saturation(
                                train_evidence["actor_input"][:1024], initial_weights
                            ),
                            "final_saturation_train_1024": first_layer_saturation(
                                train_evidence["actor_input"][:1024], final_weights
                            ),
                        },
                        "eval": {},
                    }
                    # Model never learns during evaluation; stochastic categorical original PPO.
                    model.policy.set_training_mode(False)
                    for feedback in FEEDBACK:
                        traces: list[dict[str, np.ndarray]] = []
                        episodes = []
                        for reset_seed in TEST_SEEDS:
                            world = make_world()
                            try:
                                case, numeric = test_episode(
                                    world, model, scale, feedback, model_seed,
                                    reset_seed, ledger,
                                )
                            finally:
                                world.close()
                            traces.append(numeric)
                            episodes.append(case)
                        evidence_file = output_dir / (
                            f"test_seed{model_seed}_{scale}_{feedback}.npz"
                        )
                        sha = _pack(evidence_file, traces)
                        summary["numeric_artifacts"].append({
                            "name": evidence_file.name, "sha256": sha,
                            "kind": "numeric TEST steps and paired public actor inputs",
                        })
                        details["eval"][feedback] = episodes
                    summary["results"].append(details)
                    summary["ppo_rollouts"] += TRAIN_PER_MODEL // ROLLOUT_STEPS
                finally:
                    train_env.close()
        if ledger["train"] != TRAIN_CEILING or ledger["test"] > TEST_CEILING:
            raise ValueError("Final native simulator ceiling mismatch")
        if ledger["train"] + ledger["test"] > ALL_CEILING:
            raise ValueError("Combined real native action ceiling exceeded")
        _train_seed_pair_check(summary["results"])
        comparison = score(summary["results"])
        summary["comparison"] = comparison
        summary["promotion"] = comparison["strong_feedback_improvement_both_seeds"]
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/g2_e4_p4_input_scale")
    )
    outcome = run(parser.parse_args().output_dir)
    print(
        f"G2-E4-P4 {outcome['status']} train={outcome['native_action_ledger']['train']} "
        f"test={outcome['native_action_ledger']['test']} "
        f"feedback_gate={outcome['comparison']['strong_feedback_improvement_both_seeds']}"
    )
