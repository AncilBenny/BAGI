"""Persistent episode-boundary recurrent Double DQN using TorchRL components."""
from __future__ import annotations

import random
from importlib.metadata import version
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

WIDTH, HORIZON, ACTIONS = 994, 100, 7
LEGAL = (7, 2)
FORMAT = "bagi-torchrl-episode-replay-v1"
INTERFACE = "gate-a-categorical-native-discrete-v1"
PACKAGES = ("torch", "torchrl", "tensordict")


class EpisodeValues(nn.Module):
    """Evaluate current or shifted sequence with its own exact initial history."""

    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(WIDTH, hidden, batch_first=True)
        self.head = nn.Linear(hidden, ACTIONS)

    def forward(self, observation, first, shifted, legal):
        # Batch shape (..., time); flatten only the independent episode axes.
        shape = observation.shape
        time = shape[-2]
        x = observation.reshape(-1, time, WIDTH)
        prefix = first.reshape(-1, time, WIDTH)[:, :1]
        _, (h, c) = self.lstm(prefix)
        # A shifted sequence starts AFTER the first public observation.
        # An unshifted sequence starts from zero, even with nonzero LSTM biases.
        use = shifted.reshape(-1, time, 1)[:, 0].to(h.dtype).unsqueeze(0)
        out, _ = self.lstm(x, (h * use, c * use))
        values = self.head(out).reshape(*shape[:-1], ACTIONS)
        return values.masked_fill(~legal, torch.finfo(values.dtype).min)


def inputs(observations, legal_count, *, shifted=False, first=None):
    length = observations.shape[0]
    first = observations[0] if first is None else first
    return TensorDict(
        {"observation": observations.unsqueeze(0),
         "first": first.expand(1, length, WIDTH),
         "shifted": torch.full((1, length, 1), shifted, dtype=torch.bool),
         "legal": (torch.arange(ACTIONS) < legal_count).expand(1, length, ACTIONS)},
        batch_size=[1, length],
    )


def transition_batch(row, domain):
    n = int(row["length"].item())
    obs = row["observations"][:n + 1]
    root = inputs(obs[:-1], LEGAL[domain])
    root["action"] = row["actions"][:n].unsqueeze(0)
    nxt = inputs(obs[1:], LEGAL[domain], shifted=True, first=obs[0])
    for key in ("reward", "terminated", "truncated"):
        nxt[key] = row[key][:n].reshape(1, n, 1)
    nxt["done"] = nxt["terminated"] | nxt["truncated"]
    root["next"] = nxt
    return root


class PersistentRecurrentDQN:
    """One policy/optimizer/target, separately retained whole episodes per domain."""

    def __init__(self, seed, *, capacity=256, hidden=64):
        if type(seed) is not int or type(capacity) is not int or capacity < 1:
            raise ValueError("Invalid replay configuration")
        if type(hidden) is not int or hidden < 1:
            raise ValueError("Invalid hidden size")
        self.seed, self.capacity, self.hidden_size = seed, capacity, hidden
        self.rng = random.Random(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = EpisodeValues(hidden)
        self.policy = TensorDictSequential(
            TensorDictModule(
                self.net, ["observation", "first", "shifted", "legal"], ["action_value"],
            ),
            QValueModule(action_space="categorical"),
        )
        self.loss = DQNLoss(
            self.policy, action_space="categorical", double_dqn=True, delay_value=True,
        )
        self.loss.make_value_estimator(gamma=0.99)
        self.updater = SoftUpdate(self.loss, eps=0.95)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0003)
        self.generators = [torch.Generator().manual_seed(seed + i + 1) for i in range(2)]
        self.replay = [
            TensorDictReplayBuffer(
                storage=LazyTensorStorage(capacity), batch_size=1, generator=g,
            ) for g in self.generators
        ]
        self.training_actions = self.updates = self.presentations = self.episodes = 0
        self.history = []
        self.domain = None
        self.last_samples = ()

    def begin_episode(self):
        self.history = []
        self.domain = None

    def act(self, observation, domain, *, epsilon=0.0):
        x = torch.as_tensor(observation, dtype=torch.float32).clone()
        if (type(domain) is not int or domain not in (0, 1) or x.shape != (WIDTH,)
                or not torch.isfinite(x).all() or not 0 <= epsilon <= 1):
            raise ValueError("Invalid public observation or action interface")
        if self.domain not in (None, domain) or len(self.history) >= HORIZON:
            raise ValueError("Require a new episode at domain/reset boundary")
        self.domain = domain
        self.history.append(x)
        with torch.no_grad():
            action = int(self.policy(inputs(torch.stack(self.history), LEGAL[domain]))
                         ["action"][0, -1].item())
        if epsilon and self.rng.random() < epsilon:
            action = self.rng.randrange(LEGAL[domain])
        return action

    def remember(self, domain, observations, actions, rewards, terminated, truncated):
        if self.history:
            raise ValueError("Drain episode history before storing replay")
        if type(domain) is not int or domain not in (0, 1):
            raise ValueError("Invalid domain")
        obs = torch.as_tensor(observations, dtype=torch.float32)
        a = torch.as_tensor(actions)
        r = torch.as_tensor(rewards, dtype=torch.float32)
        term, trunc = torch.as_tensor(terminated), torch.as_tensor(truncated)
        n = a.numel()
        if (not 1 <= n <= HORIZON or obs.shape != (n + 1, WIDTH)
                or any(x.shape != (n,) for x in (a, r, term, trunc))
                or a.dtype not in (torch.int32, torch.int64)
                or not ((a >= 0) & (a < LEGAL[domain])).all()
                or not torch.isfinite(obs).all() or not torch.isfinite(r).all()
                or term.dtype != torch.bool or trunc.dtype != torch.bool
                or (term[:-1] | trunc[:-1]).any() or not (term[-1] | trunc[-1])):
            raise ValueError("Invalid complete reset-to-boundary trajectory")

        def pad(x, size):
            out = torch.zeros((size, *x.shape[1:]), dtype=x.dtype)
            out[:len(x)] = x
            return out

        row = TensorDict(
            {"observations": pad(obs, HORIZON + 1), "actions": pad(a.long(), HORIZON),
             "reward": pad(r, HORIZON), "terminated": pad(term, HORIZON),
             "truncated": pad(trunc, HORIZON), "length": torch.tensor(n),
             "episode_id": torch.tensor(self.episodes)}, batch_size=[],
        )
        self.replay[domain].add(row)
        self.episodes += 1
        self.training_actions += n

    def update(self, batch_size=4):
        domains = [i for i, buffer in enumerate(self.replay) if len(buffer)]
        if not domains or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Empty replay or invalid batch size")
        if self.history:
            raise ValueError("Do not update during a live episode")
        offset = self.rng.randrange(len(domains))
        losses, ids, count = [], [], 0
        for i in range(batch_size):
            domain = domains[(offset + i) % len(domains)]
            row = self.replay[domain].sample()[0]
            ids.append((domain, int(row["episode_id"].item())))
            count += int(row["length"].item())
            losses.append(self.loss(transition_batch(row, domain))["loss"])
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite replay loss")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()
        self.updater.step()
        self.updates += 1
        self.presentations += count
        self.last_samples = tuple(ids)
        return float(loss.detach())

    def save(self, path):
        """Trusted local directory, no replacement; manifest written last."""
        if self.history:
            raise ValueError("Checkpoint only at episode boundary")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        populated = [bool(len(buffer)) for buffer in self.replay]
        for i, present in enumerate(populated):
            if present:
                sub = path / f"replay-{i}"
                sub.mkdir()
                self.replay[i].dumps(sub)
        torch.save(
            {"format": FORMAT, "interface": INTERFACE,
             "versions": {p: version(p) for p in PACKAGES},
             "seed": self.seed, "capacity": self.capacity, "hidden": self.hidden_size,
             "loss": self.loss.state_dict(), "optimizer": self.optimizer.state_dict(),
             "rng": self.rng.getstate(), "generators": [g.get_state() for g in self.generators],
             "populated": populated, "training_actions": self.training_actions,
             "updates": self.updates, "presentations": self.presentations,
             "episodes": self.episodes}, path / "state.pt",
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        d = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        if (d["format"] != FORMAT or d["interface"] != INTERFACE
                or d["versions"] != {p: version(p) for p in PACKAGES}):
            raise ValueError("Incompatible recurrent checkpoint")
        result = cls(d["seed"], capacity=d["capacity"], hidden=d["hidden"])
        result.loss.load_state_dict(d["loss"], strict=True)
        result.optimizer.load_state_dict(d["optimizer"])
        result.rng.setstate(d["rng"])
        for i, present in enumerate(d["populated"]):
            if present:
                result.replay[i].loads(path / f"replay-{i}")
            result.generators[i].set_state(d["generators"][i])
        for key in ("training_actions", "updates", "presentations", "episodes"):
            if type(d[key]) is not int or d[key] < 0:
                raise ValueError("Invalid checkpoint accounting")
            setattr(result, key, d[key])
        if result.episodes < sum(map(len, result.replay)):
            raise ValueError("Replay exceeds lifetime episode count")
        return result
