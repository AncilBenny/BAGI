"""Episode-boundary persistence for the Gate A recurrent learner.

Only tensors and primitive values are serialized; never load arbitrary pickle.
The caller supplies the public observation/action interface compatibility ID.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from bagi.agents.recurrent_replay import Episode, RecurrentReplay

FORMAT = "bagi-recurrent-replay-v1"


def save_checkpoint(learner: RecurrentReplay, path: Path, *, interface: str) -> None:
    if learner.hidden is not None or not interface:
        raise ValueError("Require an episode boundary and interface ID")
    path = Path(path)
    payload = {
        "format": FORMAT, "interface": interface,
        "width": learner.width, "actions": learner.actions, "seed": learner.seed,
        "online": learner.online.state_dict(), "target": learner.target.state_dict(),
        "optimizer": learner.optimizer.state_dict(), "rng": learner.rng.getstate(),
        "updates": learner.updates, "presentations": learner.presentations,
        "training_actions": learner.training_actions,
        "replay": [
            {
                "domain": ep.domain, "legal_actions": ep.legal_actions,
                "observations": torch.from_numpy(ep.observations.copy()),
                "actions": torch.from_numpy(ep.actions.copy()),
                "rewards": torch.from_numpy(ep.rewards.copy()),
                "terminal": torch.from_numpy(ep.terminal.copy()),
            }
            for episodes in learner.replay.values() for ep in episodes
        ],
    }
    # Publish complete bytes atomically without replacing a historical checkpoint.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, *, interface: str, width: int, actions: int) -> RecurrentReplay:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data["format"], data["interface"], data["width"], data["actions"]) != (
        FORMAT, interface, width, actions,
    ) or not interface:
        raise ValueError("Incompatible recurrent checkpoint")
    learner = RecurrentReplay(width, actions, seed=data["seed"])
    learner.online.load_state_dict(data["online"])
    learner.target.load_state_dict(data["target"])
    learner.optimizer.load_state_dict(data["optimizer"])
    learner.rng.setstate(data["rng"])
    for row in data["replay"]:
        learner.remember(Episode(
            row["observations"].numpy(), row["actions"].numpy(),
            row["rewards"].numpy(), row["terminal"].numpy(),
            row["domain"], row["legal_actions"],
        ))
    for key in ("updates", "presentations", "training_actions"):
        if type(data[key]) is not int or data[key] < 0:
            raise ValueError("Invalid checkpoint accounting")
    if data["training_actions"] < learner.training_actions:
        raise ValueError("Replay exceeds lifetime training actions")
    learner.updates = data["updates"]
    learner.presentations = data["presentations"]
    learner.training_actions = data["training_actions"]
    return learner
