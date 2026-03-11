from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np


class BasePlayer(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.player_id: Optional[int] = None
        self.renderer: Any = None
        self.last_policy_target: Optional[np.ndarray] = None

    def set_player_id(self, player_id: int) -> None:
        self.player_id = player_id

    def set_renderer(self, renderer: Any) -> None:
        self.renderer = renderer

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Sequence[int],
        info: dict[str, Any],
    ) -> int:
        raise NotImplementedError
