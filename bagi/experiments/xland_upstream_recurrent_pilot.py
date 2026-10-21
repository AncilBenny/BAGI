"""REUSE-7: run upstream XLand recurrent PPO on a frozen compositional split.

Upstream source is pinned and imported, NOT copied/reimplemented here. All
benchmark RuleSets are constructed safely in memory; no downloaded pickle.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from bagi.experiments.xland_composition_smoke import fingerprint
from bagi.experiments.xland_structural_composition import construct_ruleset

SAFE_BENCHMARK_ID = "__bagi_safe_train_primitives_only__"
UPSTREAM_COMMIT = "0d306d1414dcb6e451358d94e7d9a91879462ee1"


def create_train_benchmark():
    """Use upstream Benchmark constructor with train-only RuleSets, no I/O."""
    import jax.numpy as jnp
    from xminigrid.benchmarks import Benchmark

    train = [construct_ruleset(kind) for kind in ("hold", "near")]
    held_out = construct_ruleset("hold_then_near")
    train_hashes = {fingerprint(ruleset) for ruleset in train}
    held_hash = fingerprint(held_out)
    if len(train_hashes) != 2 or held_hash in train_hashes:
        raise AssertionError("Held-out structure leaked into training benchmark")
    if any(ruleset.rules.shape[0] != 1 for ruleset in train):
        raise AssertionError("Primitives must have exactly one rule each")
    benchmark = Benchmark(
        goals=jnp.stack([r.goal for r in train]),
        rules=jnp.stack([r.rules for r in train]),
        init_tiles=jnp.stack([r.init_tiles for r in train]),
        num_rules=jnp.ones(2, dtype=jnp.int32),
    )
    return benchmark, held_out, sorted(train_hashes), held_hash


def validate_settings(config: dict) -> tuple[int, list[int]]:
    training = config["training"]
    seeds = [int(x) for x in config["evaluation"]["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Independent evaluation seeds must be unique")
    n_envs = int(training["num_envs"])
    env_steps = int(training["steps_per_env"])
    update_steps = int(training["steps_per_update"])
    total = int(training["real_steps"])
    minibatches = int(training["num_minibatches"])
    if (n_envs < 1 or env_steps < 1 or update_steps < 1 or total < 1
            or env_steps % update_steps or total % (n_envs * env_steps)
            or n_envs % minibatches):
        raise ValueError("Training budget must be exactly divisible, positive, and minibatch-safe")
    if config["environment"] != "XLand-MiniGrid-R1-9x9":
        raise ValueError("Only reviewed 9x9 XLand environment allowed")
    if int(config["evaluation"]["episode_cap"]) < 1:
        raise ValueError("Evaluation cap must be positive")
    return total, seeds


def evaluate_existing_rollout(env, env_params, network_state, init_hstate,
                              task_rule_set, seeds: list[int]) -> dict:
    """Call upstream rollout, same regular partial observation for all tasks."""
    import jax
    from utils import rollout  # from PINNED upstream training checkout

    params = env_params.replace(ruleset=task_rule_set)
    episodes = []
    for seed in seeds:
        stats = rollout(
            jax.random.key(seed), env, params, network_state, init_hstate, 1
        )
        episodes.append({
            "seed": seed,
            "return": float(stats.reward),
            "real_evaluation_steps": int(stats.length),
            "goal_success": int(float(stats.reward) > 0),
        })
    return {
        "episodes": len(episodes),
        "successes": sum(item["goal_success"] for item in episodes),
        "real_evaluation_steps": sum(item["real_evaluation_steps"] for item in episodes),
        "details": episodes,
    }


def run(config_path: str) -> dict:
    import jax
    import xminigrid
    from flax.jax_utils import replicate, unreplicate

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    total_steps, eval_seeds = validate_settings(config)
    upstream_dir = Path(config["upstream_training_dir"]).resolve()
    if not (upstream_dir / "train_meta_task.py").is_file():
        raise FileNotFoundError("Missing pinned upstream training/ checkout")
    sys.path.insert(0, str(upstream_dir))
    try:
        from train_meta_task import TrainConfig, make_states, make_train

        benchmark, held_out, train_hashes, held_hash = create_train_benchmark()
        original_loader = xminigrid.load_benchmark

        def in_memory_only(benchmark_id):
            if benchmark_id != SAFE_BENCHMARK_ID:
                raise ValueError("External benchmark loader is disabled")
            return benchmark

        xminigrid.load_benchmark = in_memory_only
        training = config["training"]
        try:
            train_cfg = TrainConfig(
                env_id=config["environment"],
                benchmark_id=SAFE_BENCHMARK_ID,
                train_seed=int(training["seed"]),
                eval_seed=int(training["seed"]) + 19,
                total_timesteps=total_steps,
                num_envs=int(training["num_envs"]),
                num_steps_per_env=int(training["steps_per_env"]),
                num_steps_per_update=int(training["steps_per_update"]),
                update_epochs=1,
                num_minibatches=int(training["num_minibatches"]),
                rnn_hidden_dim=int(training["rnn_hidden_dim"]),
                head_hidden_dim=int(training["head_hidden_dim"]),
                eval_num_envs=int(training["num_envs"]),
                eval_num_episodes=1,
            )
            rng, env, params, returned_benchmark, init_hstate, state = make_states(train_cfg)
        finally:
            xminigrid.load_benchmark = original_loader
        if returned_benchmark.num_rulesets() != 2:
            raise AssertionError("Training unexpectedly accessed non-training task")
        params = params.replace(max_steps=int(config["evaluation"]["episode_cap"]))

        pre_training_state = state
        test_hstate = init_hstate[0][None]
        def evaluate_all(current_state):
            return {
                name: evaluate_existing_rollout(
                    env, params, current_state, test_hstate, ruleset, eval_seeds,
                )
                for name, ruleset in (
                    ("train_hold", construct_ruleset("hold")),
                    ("train_near", construct_ruleset("near")),
                    ("held_out_chain", held_out),
                )
            }

        before = evaluate_all(pre_training_state)
        t0 = time.monotonic()
        devices = jax.local_device_count()
        if devices != 1:
            raise RuntimeError("Pilot requires exactly one CPU JAX device")
        fn = make_train(env, params, returned_benchmark, train_cfg)
        trained_result = fn(
            jax.random.split(rng, devices),
            replicate(state, jax.local_devices()),
            replicate(init_hstate, jax.local_devices()),
        )
        jax.block_until_ready(trained_result)
        elapsed = time.monotonic() - t0
        trained_state = unreplicate(trained_result["state"])
        after = evaluate_all(trained_state)
        last_losses = {
            key: float(unreplicate(array)[-1])
            for key, array in trained_result["loss_info"].items()
        }
        report = {
            "question": "Does pretrained-from-scratch established recurrent PPO learn primitive skills and zero-shot compose them?",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_algorithm": "Unmodified XLand train_meta_task.py recurrent PPO and ActorCriticRNN",
            "training_real_steps": total_steps,
            "training_wall_seconds_including_jax_compilation": round(elapsed, 3),
            "model_training_seed": int(training["seed"]),
            "training_rule_hashes": train_hashes,
            "frozen_heldout_ruleset_sha256": held_hash,
            "training_rulesets": 2,
            "held_out_rulesets_in_training": 0,
            "network_observables": "Partial img, direction, prior action, prior reward; no true ruleset, full grid or goal string",
            "evaluation_seeds": eval_seeds,
            "evaluation_episode_step_cap": int(config["evaluation"]["episode_cap"]),
            "before_training": before,
            "after_training": after,
            "last_logged_loss_stats": last_losses,
            "total_evaluation_real_steps": sum(
                x["real_evaluation_steps"] for part in (before, after) for x in part.values()
            ),
            "limitations": [
                "This is one bounded CPU training seed, not an established generalization result.",
                "Sparse rewards and a very small pilot may yield zero successful training or transfer episodes.",
                "No independent memoryless-trained policy control yet; before/after is a learning check, not proof of memory benefit.",
                "The unmodified upstream meta-RL trainer evaluates training rulesets during updates; held-out ruleset is excluded from that loop.",
                "No external benchmark downloaded or deserialized, no wandb session, and no training checkpoint imported.",
                "Hidden rule/goal structure is NOT observed by the network; zero-shot chain may be unsolvable without exploration within an episode.",
            ],
        }
        dest = Path(config["output_dir"]) / "summary.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({
            "before": {k: x["successes"] for k, x in before.items()},
            "after": {k: x["successes"] for k, x in after.items()},
            "train_real_steps": total_steps, "wall_s": round(elapsed, 3),
        }, indent=2))
        return report
    finally:
        sys.path.remove(str(upstream_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_upstream_recurrent_pilot.yaml")
    run(parser.parse_args().config)
