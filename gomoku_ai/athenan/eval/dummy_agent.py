"""Dummy Athenan agent used for interface smoke tests."""

from __future__ import annotations

import numpy as np

from gomoku_ai.common.agents import BaseAgent
from gomoku_ai.env import GomokuEnv


class AthenanDummyAgent(BaseAgent):
    """Minimal Athenan placeholder agent.

    The agent intentionally uses no search/network logic yet and selects one
    legal move directly from the environment.
    """

    def __init__(self, *, pick_mode: str = "first_legal", seed: int | None = None) -> None:
        normalized_mode = pick_mode.strip().lower()
        if normalized_mode not in {"first_legal", "random_legal"}:
            raise ValueError("pick_mode must be one of {'first_legal', 'random_legal'}.")
        self.pick_mode = normalized_mode
        self._rng = np.random.default_rng(seed)

    def select_action(self, env: GomokuEnv) -> int:
        """Return one legal action from the current `GomokuEnv` position."""

        legal_actions = np.flatnonzero(np.asarray(env.get_valid_moves(), dtype=bool))
        if legal_actions.size == 0:
            raise RuntimeError("No legal moves are available in the current state.")

        if self.pick_mode == "random_legal":
            return int(self._rng.choice(legal_actions))
        return int(legal_actions[0])
