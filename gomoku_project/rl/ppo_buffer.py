from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PPOTransition:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    advantage: float = 0.0
    return_target: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


class PPOBuffer:
    def __init__(self) -> None:
        self._storage: list[PPOTransition] = []
        self._episode_steps: list[PPOTransition] = []
        self.episodes_collected = 0

    def add(
        self,
        *,
        observation: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        self._episode_steps.append(
            PPOTransition(
                observation=np.asarray(observation, dtype=np.int8).copy(),
                action_mask=np.asarray(action_mask, dtype=bool).copy(),
                action=int(action),
                log_prob=float(log_prob),
                value=float(value),
                reward=float(reward),
                done=bool(done),
                info=dict(info or {}),
            )
        )

    def finish_episode(self, *, gamma: float, gae_lambda: float, use_gae: bool = True) -> int:
        if not self._episode_steps:
            return 0

        rewards = np.asarray([step.reward for step in self._episode_steps], dtype=np.float32)
        values = np.asarray([step.value for step in self._episode_steps], dtype=np.float32)
        dones = np.asarray([step.done for step in self._episode_steps], dtype=np.float32)

        returns = np.zeros_like(rewards, dtype=np.float32)
        future_return = 0.0
        for index in range(len(self._episode_steps) - 1, -1, -1):
            next_non_terminal = 0.0 if dones[index] else 1.0
            future_return = rewards[index] + gamma * future_return * next_non_terminal
            returns[index] = future_return

        if use_gae:
            advantages = np.zeros_like(rewards, dtype=np.float32)
            gae = 0.0
            for index in range(len(self._episode_steps) - 1, -1, -1):
                next_non_terminal = 0.0 if dones[index] else 1.0
                next_value = values[index + 1] if index + 1 < len(values) and not dones[index] else 0.0
                delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]
                gae = delta + gamma * gae_lambda * next_non_terminal * gae
                advantages[index] = gae
        else:
            advantages = returns - values

        for step, advantage, return_target in zip(self._episode_steps, advantages, returns):
            step.advantage = float(advantage)
            step.return_target = float(return_target)

        self._storage.extend(self._episode_steps)
        episode_length = len(self._episode_steps)
        self._episode_steps = []
        self.episodes_collected += 1
        return episode_length

    def as_batch(self) -> dict[str, np.ndarray]:
        if not self._storage:
            raise ValueError("PPOBuffer is empty.")

        return {
            "observations": np.stack([step.observation for step in self._storage]).astype(np.int8),
            "action_masks": np.stack([step.action_mask for step in self._storage]).astype(bool),
            "actions": np.asarray([step.action for step in self._storage], dtype=np.int64),
            "log_probs": np.asarray([step.log_prob for step in self._storage], dtype=np.float32),
            "values": np.asarray([step.value for step in self._storage], dtype=np.float32),
            "rewards": np.asarray([step.reward for step in self._storage], dtype=np.float32),
            "advantages": np.asarray([step.advantage for step in self._storage], dtype=np.float32),
            "returns": np.asarray([step.return_target for step in self._storage], dtype=np.float32),
        }

    def clear(self) -> None:
        self._storage = []
        self._episode_steps = []
        self.episodes_collected = 0

    def __len__(self) -> int:
        return len(self._storage)
