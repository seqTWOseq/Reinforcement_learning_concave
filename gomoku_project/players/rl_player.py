from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np

from gomoku_project.core.game import GomokuGame
from gomoku_project.players.base import BasePlayer
from gomoku_project.rl.mcts import AlphaZeroMCTS

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency fallback
    torch = None


class RLPlayer(BasePlayer):
    def __init__(
        self,
        model: Any = None,
        device: str = "cpu",
        deterministic: bool = True,
        name: str = "RLPlayer",
        num_simulations: int = 32,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        use_root_noise: bool = False,
        temperature_drop_move: int = 10,
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_root_noise = use_root_noise
        self.temperature_drop_move = temperature_drop_move
        self.move_count = 0

        if self.model is not None and torch is not None and hasattr(self.model, "to"):
            self.model.to(self.device)
            self.model.eval()

    def reset(self) -> None:
        self.last_policy_target = None
        self.move_count = 0
        if self.model is not None and torch is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Sequence[int],
        info: dict[str, Any],
    ) -> int:
        if not valid_actions:
            return 0

        if self.model is None or torch is None:
            action = int(random.choice(list(valid_actions)))
            self.last_policy_target = self._one_hot(action, observation.shape[0] * observation.shape[0])
            return action

        game = GomokuGame.from_state(
            board=np.asarray(info["board"], dtype=np.int8),
            current_player=int(info["current_player"]),
            done=bool(info.get("done", False)),
            winner=info.get("winner"),
            last_action=info.get("last_action"),
            last_reason=str(info.get("reason", "manual")),
        )
        mcts = AlphaZeroMCTS(
            self.model,
            device=self.device,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )
        visit_probs = mcts.run(
            game,
            add_exploration_noise=self.use_root_noise and not self.deterministic,
        )
        self.last_policy_target = visit_probs.copy()
        action = self._sample_action(visit_probs)
        self.move_count += 1
        return int(action)

    def _sample_action(self, visit_probs: np.ndarray) -> int:
        if self.deterministic or self.move_count >= self.temperature_drop_move:
            return int(np.argmax(visit_probs))

        total = float(visit_probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            return int(np.argmax(visit_probs))

        probabilities = visit_probs / total
        return int(np.random.choice(len(probabilities), p=probabilities))

    @staticmethod
    def _one_hot(action: int, action_space_size: int) -> np.ndarray:
        target = np.zeros(action_space_size, dtype=np.float32)
        target[int(action)] = 1.0
        return target


AlphaZeroPlayer = RLPlayer
