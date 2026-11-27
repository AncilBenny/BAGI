"""Exploratory single-family competence diagnostic, not transfer evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from bagi.agents.dqn import Transition
from bagi.agents.persistent_dqn import RuntimeDQN
from bagi.agents.recurrent_checkpoint import load_checkpoint, save_checkpoint
from bagi.agents.recurrent_replay import RecurrentReplay
from bagi.experiments.cumulative_transfer import (
    ACTIONS,
    HORIZON,
    TASKS,
    WIDTH,
    encode,
    make_env,
    run_episode,
)
from bagi.workshop_runtime import source_revision

ARMS = ("recurrent_original", "recurrent_dense", "existing_dqn", "recurrent_ppo")
BLOCK = 256
INTERFACE = "gate-a-public-interface-v1"


class DQNControl:
    """Reuse frozen DQN; compensate its fixed /6 input normalization."""

    def __init__(self, legal, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.agent = RuntimeDQN((1, WIDTH), legal, 0.0003, 0.99, seed=seed)
        self.training_actions = 0
        self.presentations = 0

    @property
    def updates(self):
        return self.agent.optimizer_updates

    def begin_episode(self):
        pass

    def act(self, observation, legal_actions, *, epsilon):
        if legal_actions != self.agent.actions:
            raise ValueError("Wrong action interface")
        obs = observation.reshape(1, WIDTH) * 6
        if epsilon:
            return self.agent.select_action(obs, epsilon)
        with torch.no_grad():
            return int(self.agent.online(torch.as_tensor(obs).unsqueeze(0)).argmax().item())

    def remember(self, episode):
        for i, action in enumerate(episode.actions):
            self.agent.buffer.add(Transition(
                episode.observations[i].reshape(1, WIDTH) * 6, int(action),
                float(episode.rewards[i]), episode.observations[i + 1].reshape(1, WIDTH) * 6,
                bool(episode.terminal[i]),
            ))
        self.training_actions += len(episode.actions)

    def update(self):
        loss = self.agent.learn(64)
        if loss is None or not np.isfinite(loss):
            raise RuntimeError("Missing or nonfinite DQN update")
        self.presentations += 64
        if self.updates % 20 == 0:
            self.agent.sync_target()


class NativePPOEnv(gym.Wrapper):
    """Identical public encoding, reset sequence and block cuts; log real steps."""

    def __init__(self, domain, seed_base):
        super().__init__(make_env(domain))
        self.domain = domain
        self.seed_base = seed_base
        self.observation_space = gym.spaces.Box(-1, 1, (WIDTH,), dtype=np.float32)
        self.rows = []
        self.steps = 0
        self.episodes = 0
        self.episode_actions = []

    def reset(self, *, seed=None, options=None):
        self.current_seed = self.seed_base + self.episodes
        self.episodes += 1
        self.episode_actions = []
        self.reward_sum = 0.0
        obs, _ = self.env.reset(seed=self.current_seed, options=options)
        return encode(self.domain, obs), {}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Illegal PPO action")
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.steps += 1
        self.episode_actions.append(action)
        self.reward_sum += float(reward)
        native_end = bool(terminated or truncated)
        cut = self.steps % BLOCK == 0 and not native_end
        if native_end or cut:
            if self.domain == TASKS[0]:
                native = self.env.unwrapped
                tile = native.grid.get(*native.agent_pos)
                success = bool(terminated and tile is not None and tile.type == "goal")
            else:
                success = bool(truncated and not terminated
                               and len(self.episode_actions) == HORIZON)
            self.rows.append({
                "domain": self.domain, "seed": self.current_seed,
                "actions": len(self.episode_actions), "executed_actions": self.episode_actions,
                "action_sha256": hashlib.sha256(bytes(self.episode_actions)).hexdigest(),
                "reward": self.reward_sum, "physical_success": success,
                "native_end": native_end, "learn": True,
            })
        done = native_end or cut
        return (encode(self.domain, obs, action, reward, done), reward,
                terminated, truncated or cut, {})


class PPOControl:
    def __init__(self, domain, seed, seed_base):
        from sb3_contrib import RecurrentPPO

        self.env = NativePPOEnv(domain, seed_base)
        self.model = RecurrentPPO(
            "MlpLstmPolicy", self.env, seed=seed, device="cpu", n_steps=BLOCK,
            batch_size=64, n_epochs=4, target_kl=None,
            policy_kwargs={"lstm_hidden_size": 64, "net_arch": {"pi": [32], "vf": [32]}},
            verbose=0,
        )
        self.updates = 0
        self.presentations = 0
        self.hidden = None
        self.model.policy.optimizer.register_step_post_hook(self._count_update)
        original = self.model.rollout_buffer.get

        def counted_get(*args, **kwargs):
            for batch in original(*args, **kwargs):
                before = self.updates
                yield batch
                if self.updates != before + 1:
                    raise RuntimeError("Expected one actual PPO optimizer step per minibatch")
                self.presentations += int((batch.mask > 1e-8).sum().item())

        self.model.rollout_buffer.get = counted_get

    def _count_update(self, optimizer, args, kwargs):
        self.updates += 1

    @property
    def training_actions(self):
        return self.env.steps

    def begin_episode(self):
        self.hidden = None

    def act(self, observation, legal_actions, *, epsilon):
        if epsilon:
            raise ValueError("PPO wrapper is for frozen evaluation only")
        action, self.hidden = self.model.predict(
            observation, state=self.hidden,
            episode_start=np.array([self.hidden is None]), deterministic=True,
        )
        action = int(np.asarray(action).item())
        if not 0 <= action < legal_actions:
            raise ValueError("Invalid PPO action")
        return action

    def train(self, steps):
        before = self.env.steps
        self.model.learn(total_timesteps=steps, reset_num_timesteps=False)
        if self.env.steps - before != steps or self.model.num_timesteps != self.env.steps:
            raise RuntimeError("PPO native action accounting mismatch")


def frozen_evaluation(learner, domain, seeds):
    before = (learner.training_actions, learner.updates, learner.presentations)
    rows = [run_episode(learner, domain, seed, learn=False) for seed in seeds]
    if before != (learner.training_actions, learner.updates, learner.presentations):
        raise RuntimeError("Evaluation changed learner counters")
    return {
        "success": sum(row["physical_success"] for row in rows) / len(rows),
        "mean_length": sum(row["actions"] for row in rows) / len(rows),
        "actions": sum(row["actions"] for row in rows), "episodes": rows,
    }


def run(output, *, domain=TASKS[1], seeds=(71, 73, 79), budget=32768,
        interval=4096, eval_count=20, arms=ARMS):
    if (domain not in TASKS or not seeds or len(set(seeds)) != len(seeds)
            or budget <= 0 or interval <= 0 or budget % interval or interval % BLOCK
            or budget >= 100000 or not 1 <= eval_count < 100000
            or not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS)):
        raise ValueError("Invalid exploratory diagnostic configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.monotonic()
    results = []
    with (output / "training_actions.jsonl").open("x") as trace:
        for index, seed in enumerate(seeds):
            base = 11_000_000 + TASKS.index(domain) * 10_000_000 + index * 1_000_000
            eval_seeds = range(base + 500_000, base + 500_000 + eval_count)
            for arm in arms:
                began = time.monotonic()
                probe = make_env(domain)
                legal = int(probe.action_space.n)
                probe.close()
                if arm == "existing_dqn":
                    learner = DQNControl(legal, seed)
                elif arm == "recurrent_ppo":
                    learner = PPOControl(domain, seed, base)
                else:
                    learner = RecurrentReplay(WIDTH, ACTIONS, seed=seed)
                offset = 0
                curve = []
                checkpoint = None
                try:
                    for steps in range(0, budget + 1, interval):
                        point = frozen_evaluation(learner, domain, eval_seeds)
                        curve.append({"train_actions": steps, **point})
                        if steps == budget:
                            break
                        if steps == budget // 2 and isinstance(learner, RecurrentReplay):
                            path = output / f"{arm}-{seed}.pt"
                            save_checkpoint(learner, path, interface=INTERFACE)
                            learner = load_checkpoint(
                                path, interface=INTERFACE, width=WIDTH, actions=ACTIONS,
                            )
                            with path.open("rb") as handle:
                                checkpoint = hashlib.file_digest(handle, "sha256").hexdigest()
                            path.unlink()
                        if arm == "recurrent_ppo":
                            learner.train(interval)
                            rows = learner.env.rows
                            learner.env.rows = []
                        else:
                            rows = []
                            updates = 4 if arm == "recurrent_original" else 32
                            for _ in range(interval // BLOCK):
                                remaining = BLOCK
                                while remaining:
                                    row = run_episode(
                                        learner, domain, base + offset,
                                        learn=True, limit=min(remaining, HORIZON),
                                    )
                                    offset += 1
                                    remaining -= row["actions"]
                                    rows.append(row)
                                for _ in range(updates):
                                    learner.update()
                        if sum(row["actions"] for row in rows) != interval:
                            raise RuntimeError("Native training trace budget mismatch")
                        if learner.training_actions != steps + interval:
                            raise RuntimeError("Training interaction counter mismatch")
                        for row in rows:
                            trace.write(json.dumps(dict(row, arm=arm, model_seed=seed)) + "\n")
                    scores = [p["success"] for p in curve]
                    results.append({
                        "arm": arm, "seed": seed, "curve": curve,
                        "success_auc": (sum((a + b) / 2 for a, b in pairwise(scores))
                                        / (len(scores) - 1)),
                        "training_actions": learner.training_actions,
                        "optimizer_updates": learner.updates,
                        "sample_presentations": learner.presentations,
                        "evaluation_actions": sum(p["actions"] for p in curve),
                        "resumed_checkpoint_sha256": checkpoint,
                        "seconds": time.monotonic() - began,
                    })
                finally:
                    if arm == "recurrent_ppo":
                        learner.env.close()
    summary = {
        "kind": "gate_a_single_family_competence_diagnostic", "scientific_pass": False,
        "source": source_revision(),
        "versions": {p: version(p) for p in (
            "gymnasium", "minigrid", "torch", "numpy", "stable-baselines3", "sb3-contrib",
        )},
        "config": {"domain": domain, "seeds": seeds, "budget": budget, "interval": interval,
                   "eval_count": eval_count, "arms": arms, "block": BLOCK},
        "actual_training_actions": sum(r["training_actions"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "actual_optimizer_updates": sum(r["optimizer_updates"] for r in results),
        "actual_sample_presentations": sum(r["sample_presentations"] for r in results),
        "seconds": time.monotonic() - start, "results": results,
        "limitations": "Exploratory monitor only; no transfer or confirmatory inference.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--domain", choices=TASKS, default=TASKS[1])
    args = parser.parse_args()
    print(json.dumps(run(args.output, domain=args.domain), indent=2))
