from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gomoku_project.core.constants import BOARD_SIZE
from gomoku_project.core.game import GomokuGame
from gomoku_project.core.utils import board_to_ansi


class GomokuEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, board_size: int = BOARD_SIZE, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode
        self.game = GomokuGame(board_size=self.board_size)

        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(self.board_size, self.board_size),
            dtype=np.int8,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.game.reset()
        observation = self.game.get_observation()
        info = self.game.get_info()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.game.step(int(action))
        observation = self.game.get_observation(player=self.game.current_player)
        info = self.game.get_info()
        terminated = result.done
        truncated = False
        return observation, result.reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        return board_to_ansi(self.game.board)

    def close(self) -> None:
        return None
