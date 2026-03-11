from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from gomoku_project.players.base import BasePlayer


class CenterPlayer(BasePlayer):
    def __init__(self, name: str = "CenterPlayer") -> None:
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

        board_size = observation.shape[0]
        center = board_size // 2
        return int(
            min(
                valid_actions,
                key=lambda action: abs(action // board_size - center) + abs(action % board_size - center),
            )
        )

