from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AlphaZeroExample:
    observation: np.ndarray
    policy_target: np.ndarray
    value_target: float
    info: dict[str, Any]


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._buffer: deque[AlphaZeroExample] = deque(maxlen=capacity)

    def add(self, example: AlphaZeroExample) -> None:
        self._buffer.append(example)

    def extend(self, examples: list[AlphaZeroExample]) -> None:
        self._buffer.extend(examples)

    def sample(self, batch_size: int) -> list[AlphaZeroExample]:
        batch_size = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
