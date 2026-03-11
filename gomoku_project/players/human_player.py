from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from gomoku_project.players.base import BasePlayer


class HumanPlayer(BasePlayer):
    def __init__(self, renderer: Any = None, name: str = "Human") -> None:
        super().__init__(name=name)
        self.renderer = renderer

    def reset(self) -> None:
        return None

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Sequence[int],
        info: dict[str, Any],
    ) -> int:
        if self.renderer is None:
            raise RuntimeError("HumanPlayer requires a TkinterRenderer instance.")

        board = info.get("board")
        self.renderer.render(
            board=board,
            current_player=info.get("current_player"),
            last_action=info.get("last_action"),
        )
        return self.renderer.wait_for_action(valid_actions)

