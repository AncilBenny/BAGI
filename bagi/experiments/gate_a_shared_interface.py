"""Common native interface and rollout-boundary continuation for maintained PPO."""
from __future__ import annotations

import hashlib
import random
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_competence import BLOCK, NativePPOEnv, PPOControl
from bagi.experiments.gate_a_representation import CategoricalView
from bagi.workshop_runtime import source_revision

INTERFACE = "gate-a-categorical-two-actuators-v1"
IMAGE, WIDTH = 980, 994
PACKAGES = ("gymnasium", "minigrid", "numpy", "torch", "stable-baselines3", "sb3-contrib")


def native_action(domain, proposal):
    a = np.asarray(proposal)
    if (domain not in TASKS or a.shape != (2,) or not np.issubdtype(a.dtype, np.integer)
            or not 0 <= a[0] < 7 or not 0 <= a[1] < 2):
        raise ValueError("Invalid actuator proposal")
    return int(a[TASKS.index(domain)])


class SharedActuators(gym.Wrapper):
    """Only the declared active channel reaches the native environment."""

    def __init__(self, env):
        super().__init__(env)
        self.domain = env.domain
        self.view = CategoricalView(env) if self.domain == TASKS[0] else None
        self.action_space = gym.spaces.MultiDiscrete([7, 2])
        self.observation_space = gym.spaces.Box(-1, 1, (WIDTH,), dtype=np.float32)
        if self.view is not None and self.view.observation_space != self.observation_space:
            raise ValueError("Unexpected categorical observation width")
        self.proposals = []

    def encode(self, scalar):
        if self.view is not None:
            return self.view.observation(scalar)
        return np.concatenate((np.zeros(IMAGE, dtype=np.float32), scalar[147:]))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.proposals = []
        return self.encode(obs), info

    def step(self, proposal):
        action = native_action(self.domain, proposal)
        self.proposals.append(np.asarray(proposal).tolist())
        obs, reward, term, trunc, info = self.env.step(action)
        if term or trunc:
            row = self.env.rows[-1]
            if row["executed_actions"] != [native_action(self.domain, p) for p in self.proposals]:
                raise RuntimeError("Joint proposal/native execution mismatch")
            row["actuator_proposals"] = self.proposals.copy()
        return self.encode(obs), reward, term, trunc, info


class SharedPPO(PPOControl):
    def __init__(self, domain, seed, seed_base):
        super().__init__(domain, seed, seed_base, observation_wrapper=SharedActuators)

    def switch_domain(self, domain, seed_base):
        check_boundary(self)
        previous = self.model_env
        replacement = NativePPOEnv(domain, seed_base)
        replacement.steps = self.training_actions
        wrapped = SharedActuators(replacement)
        self.model.set_env(wrapped, force_reset=True)
        state = self.model._last_lstm_states
        self.model._last_lstm_states = type(state)(
            tuple(torch.zeros_like(x) for x in state.pi),
            tuple(torch.zeros_like(x) for x in state.vf),
        )
        self.env, self.model_env = replacement, wrapped
        previous.close()


def check_boundary(control):
    env, model = control.env, control.model
    if (not env.steps or env.steps % BLOCK or env.rows or env.episode_actions
            or model.num_timesteps != env.steps or model._last_obs is None
            or not np.asarray(model._last_episode_starts).all()
            or control.hidden is not None or env.episodes < 1
            or env.current_seed != env.seed_base + env.episodes - 1):
        raise ValueError("Require completed rollout and drained trace at pending reset")


def capture_rng():
    kind, keys, pos, gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (kind, keys.tolist(), pos, gauss, cached),
            "torch": torch.random.get_rng_state().clone()}


def restore_rng(state):
    py = random.Random()
    py.setstate(state["python"])
    kind, keys, pos, gauss, cached = state["numpy"]
    if (kind != "MT19937" or len(keys) != 624
            or any(type(k) is not int or not 0 <= k < 2**32 for k in keys)
            or not 0 <= pos <= 624 or gauss not in (0, 1) or not np.isfinite(cached)):
        raise ValueError("Invalid NumPy RNG")
    nr = np.random.RandomState()
    nr.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, gauss, cached))
    tr = torch.Generator(device="cpu")
    tr.set_state(state["torch"])
    random.setstate(py.getstate())
    np.random.set_state(nr.get_state())
    torch.random.set_rng_state(tr.get_state())


def save_boundary(control, path):
    """Single CPU environment, completed rollouts only; no pickled model objects."""
    check_boundary(control)
    env, model = control.env, control.model
    state = model._last_lstm_states
    data = {
        "interface": INTERFACE, "versions": {p: version(p) for p in PACKAGES},
        "source": source_revision(), "domain": env.domain, "seed_base": env.seed_base,
        "seed": model.seed, "reset_cursor": env.episodes, "native_steps": env.steps,
        "updates": control.updates, "presentations": control.presentations,
        "model_updates": model._n_updates, "episode_num": model._episode_num,
        "policy": model.policy.state_dict(), "optimizer": model.policy.optimizer.state_dict(),
        "pending_observation": torch.as_tensor(model._last_obs.copy()),
        "actor_lstm": tuple(x.detach().clone() for x in state.pi),
        "critic_lstm": tuple(x.detach().clone() for x in state.vf),
        "rng": capture_rng(),
    }
    path = Path(path)
    with path.open("xb") as handle:
        torch.save(data, handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_boundary(path, digest, *, expected_source=None):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Checkpoint digest mismatch")
    d = torch.load(path, map_location="cpu", weights_only=True)
    expected = source_revision() if expected_source is None else expected_source
    if (d["interface"] != INTERFACE or d["versions"] != {p: version(p) for p in PACKAGES}
            or d["source"] != expected or expected["tracked_worktree_dirty"] is not False):
        raise ValueError("Incompatible checkpoint provenance")
    counts = ("native_steps", "reset_cursor", "updates", "presentations",
              "model_updates", "episode_num")
    if (any(type(d[k]) is not int or d[k] < 0 for k in counts)
            or not d["native_steps"] or d["native_steps"] % BLOCK
            or not 1 <= d["reset_cursor"] <= d["native_steps"] + 1
            or not d["updates"] or d["presentations"] < d["native_steps"]):
        raise ValueError("Invalid checkpoint counters")
    control = SharedPPO(d["domain"], d["seed"], d["seed_base"])
    model, env = control.model, control.env
    try:
        env.episodes = d["reset_cursor"] - 1
        obs = model.get_env().reset()
        if not np.array_equal(obs, d["pending_observation"].numpy()):
            raise ValueError("Pending reset observation mismatch")
        env.steps = d["native_steps"]
        control.updates, control.presentations = d["updates"], d["presentations"]
        model.num_timesteps = d["native_steps"]
        model._n_updates, model._episode_num = d["model_updates"], d["episode_num"]
        model._last_obs = obs
        model._last_episode_starts = np.ones((1,), dtype=bool)
        model.policy.load_state_dict(d["policy"], strict=True)
        model.policy.optimizer.load_state_dict(d["optimizer"])
        model._last_lstm_states = type(model._last_lstm_states)(d["actor_lstm"], d["critic_lstm"])
        check_boundary(control)
        restore_rng(d["rng"])
        return control
    except Exception:
        control.model_env.close()
        raise
