"""Native Workshop diagnostic using BAGI's shared runtime and persistent learner.

Workshop is a diagnostic, not a definition of intelligence. The policy sees
only the local observation and the learner sees actual actions and rewards.
The separate outcome audit may use native terminal events; never feed those
events or hidden simulator information into the policy or replay.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bagi.agents.general_brain import FactChange, GeneralBrain, Goal, Skill
from bagi.agents.persistent_dqn import PersistentDQN
from bagi.runtime import AgentRuntime, StepFeedback
from bagi.worlds.workshop import WorkshopEnv

COMPATIBILITY = "workshop-v02-public-local-5x5-cardinal4-dqn-v1"
GOAL = Goal("workshop", frozenset({"positive_reward_observed"}))


class RewardSensor:
    """Track public reward only, without claiming a physical goal from it."""

    def initial(self, observation):
        return frozenset()

    def change(self, before, executed_action, after, public_reward):
        return FactChange(
            frozenset({"positive_reward_observed"}) if public_reward > 0 else frozenset()
        )


def make_runtime(learner: PersistentDQN, *, max_steps: int, learn: bool) -> AgentRuntime:
    if learner.config["observation_shape"] != (5, 5) or learner.config["actions"] != 4:
        raise ValueError("Incompatible Workshop observation/action space")
    skill = Skill(
        "navigate", frozenset(), GOAL.required,
        learner.policy if learn else learner.greedy, max_actions=max_steps,
    )
    brain = GeneralBrain(
        (skill,), RewardSensor(), max_episode_actions=max_steps, max_attempts_per_skill=1
    )
    return AgentRuntime(brain, COMPATIBILITY)


def native_episode(
    runtime: AgentRuntime, learner: PersistentDQN, env: WorkshopEnv, *, seed: int, learn: bool
) -> dict:
    """Run native actions; separately audit terminal success from environment events."""
    observation, _ = env.reset(seed=seed)
    actual = []
    reward_sum = 0.0
    physical = False
    steps_before = learner.training_actions
    updates_before = learner.agent.optimizer_updates

    def step(action):
        nonlocal reward_sum, physical
        if type(action) is not int or not env.action_space.contains(action):
            raise ValueError("Invalid Workshop action")
        after, reward, terminated, truncated, info = env.step(action)
        actual.append(action)
        reward_sum += float(reward)
        physical = bool(terminated and "goal_reached" in info["events"])
        return StepFeedback(action, after, float(reward), bool(terminated), bool(truncated))

    result = runtime.run_episode(
        GOAL, observation, step, learn=learn, on_transition=learner.observe
    )
    if tuple(actual) != result.actions:
        raise RuntimeError("Native action accounting differs from runtime")
    training_actions = learner.training_actions - steps_before
    updates = learner.agent.optimizer_updates - updates_before
    if training_actions != (len(actual) if learn else 0) or (not learn and updates):
        raise RuntimeError("Learner action/update accounting mismatch")
    return {
        "seed": seed,
        "native_actions": len(actual),
        "executed_actions": actual,
        "training_actions": training_actions,
        "optimizer_updates": updates,
        "reward": reward_sum,
        "public_reward_observed": result.goal_verified,
        "physical_success": physical,
        "reason": result.reason,
    }


def source_revision() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip())
        return {"commit": sha, "tracked_worktree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_worktree_dirty": None}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_workshop(
    output: Path, *, mode: str, episodes: int, first_seed: int, model_seed: int = 0,
    max_steps: int = 100, batch_size: int = 64, epsilon: float = 0.2,
    layout: str = "default", checkpoint: Path | None = None,
) -> dict:
    """Run a new isolated diagnostic. No scientific split or transfer claim is implied."""
    if mode not in {"train", "evaluate"} or episodes < 1 or max_steps < 1 or first_seed < 0:
        raise ValueError("Invalid diagnostic settings")
    learn = mode == "train"
    if not learn and checkpoint is None:
        raise ValueError("Evaluation requires a checkpoint")
    if checkpoint is None:
        learner = PersistentDQN((5, 5), 4, seed=model_seed, batch_size=batch_size, epsilon=epsilon)
        state = None
        parent_digest = None
    else:
        parent_digest = digest(checkpoint)
        learner, state = PersistentDQN.load(checkpoint, compatibility_id=COMPATIBILITY)
    runtime = make_runtime(learner, max_steps=max_steps, learn=learn)
    if state is not None:
        runtime.restore_snapshot(state)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = WorkshopEnv(
        view_size=5, max_steps=max_steps, layout_mode=layout, start_mode="seeded_start_chamber"
    )
    rows = []
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as trace:
            for seed in range(first_seed, first_seed + episodes):
                row = native_episode(runtime, learner, env, seed=seed, learn=learn)
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                rows.append(row)
    finally:
        env.close()
    saved = None
    if learn:
        path = output / "learner.pt"
        learner.save(path, runtime)
        saved = digest(path)
    summary = {
        "kind": "native_workshop_diagnostic",
        "scientific_transfer_claim": False,
        "mode": mode,
        "source": source_revision(),
        "config": learner.config,
        "compatibility_id": COMPATIBILITY,
        "layout": layout,
        "start_mode": "seeded_start_chamber",
        "max_steps": max_steps,
        "first_seed": first_seed,
        "episodes": len(rows),
        "native_actions": sum(r["native_actions"] for r in rows),
        "training_actions": sum(r["training_actions"] for r in rows),
        "optimizer_updates": sum(r["optimizer_updates"] for r in rows),
        "physical_successes": sum(r["physical_success"] for r in rows),
        "reward": sum(r["reward"] for r in rows),
        "parent_checkpoint_sha256": parent_digest,
        "checkpoint_sha256": saved,
        "lifetime_training_actions": learner.training_actions,
        "lifetime_optimizer_updates": learner.agent.optimizer_updates,
        "limitations": "Diagnostic only; caller must register independent splits for research.",
    }
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary
