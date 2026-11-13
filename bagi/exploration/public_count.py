"""Task-neutral episodic first-visit intrinsic-reward reference baseline.

This is the established *count-based exploration* principle, not a new learning
algorithm. It depends exclusively on observations the acting policy can see.
Keep intrinsic rewards separate from native extrinsic goal rewards in reports.
"""

from __future__ import annotations

import math

import numpy as np


class EpisodicFirstVisit:
    """Pay a fixed bonus once per *exact public observation* per episode.

    Counts local observations, not true physical states. Distinct physical
    states may alias, and this baseline does not supply memory to a policy.
    No simulator action, reward, private info or reset is performed here.
    """

    def __init__(self, bonus: float = 0.25):
        if not math.isfinite(bonus) or bonus < 0:
            raise ValueError("bonus must be finite and nonnegative")
        self.bonus = float(bonus)
        self._seen: set[bytes] = set()
        self._shape: tuple[int, ...] | None = None
        self._dtype: str | None = None

    def _key(self, observation: np.ndarray) -> bytes:
        array = np.asarray(observation)
        if array.ndim < 1 or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("public observation must be a finite numeric array")
        shape, dtype = tuple(array.shape), array.dtype.str
        if self._shape is None:
            raise RuntimeError("reset with the initial observation before observe")
        if shape != self._shape or dtype != self._dtype:
            raise ValueError("public observation shape or dtype changed within episode")
        return np.ascontiguousarray(array).tobytes()

    def reset(self, initial_public_observation: np.ndarray) -> None:
        """Reset at the real episode boundary, recording the starting view."""
        array = np.asarray(initial_public_observation)
        if array.ndim < 1 or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("public observation must be a finite numeric array")
        self._shape = tuple(array.shape)
        self._dtype = array.dtype.str
        self._seen = {self._key(array)}

    def observe(self, next_public_observation: np.ndarray) -> float:
        """Return intrinsic-only reward after a *real* executed action."""
        key = self._key(next_public_observation)
        if key in self._seen:
            return 0.0
        self._seen.add(key)
        return self.bonus

    @property
    def unique_public_views(self) -> int:
        return len(self._seen)
