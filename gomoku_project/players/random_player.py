from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from gomoku_project.players.base import BasePlayer


class RandomPlayer(BasePlayer):
    def __init__(self, name: str = "RandomPlayer") -> None:
        super().__init__(name=name)

    def reset(self) -> None:
        return None

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Sequence[int],
        info: dict[str, Any],
    ) -> int:
        if not valid_actions:
            return 0
        return int(np.random.choice(valid_actions))

