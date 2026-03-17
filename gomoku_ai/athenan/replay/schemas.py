"""Replay sample schema for Athenan search-to-training data flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from gomoku_ai.athenan.utils import ATHENAN_FEATURE_DTYPE, ATHENAN_FEATURE_PLANES
from gomoku_ai.env import BLACK, WHITE

ATHENAN_REPLAY_STATE_PLANES = ATHENAN_FEATURE_PLANES
ATHENAN_REPLAY_STATE_DTYPE = ATHENAN_FEATURE_DTYPE


@dataclass(frozen=True)
class AthenanReplaySample:
    """One replay sample generated from one root search call.

    State contract:
    - `state` is encoded planes from `encode_env_to_planes(env)`
    - shape `(3, board_size, board_size)` with `np.float32` dtype
    - input arrays are normalized into a float32 copy, then validated
    - `best_action` means the action actually played in the trajectory
    """

    state: np.ndarray
    player_to_move: int
    best_action: int
    searched_value: float
    action_values: dict[int, float]
    principal_variation: list[int]
    nodes: int
    depth_reached: int
    forced_tactical: bool
    final_outcome: float | None = None

    def __post_init__(self) -> None:
        # Normalize to float32 while still enforcing strict shape/finiteness checks.
        normalized_state = np.asarray(self.state, dtype=np.float32)
        if normalized_state.ndim != 3:
            raise ValueError(f"state must have shape (C, H, W), got ndim={normalized_state.ndim}.")
        if normalized_state.shape[0] != ATHENAN_REPLAY_STATE_PLANES:
            raise ValueError(
                f"state channel dimension must be {ATHENAN_REPLAY_STATE_PLANES}, got {normalized_state.shape[0]}."
            )
        if normalized_state.shape[1] <= 0 or normalized_state.shape[2] <= 0:
            raise ValueError(f"state spatial dimensions must be positive, got {normalized_state.shape[1:]}.")
        if not np.isfinite(normalized_state).all():
            raise ValueError("state must contain only finite values.")
        object.__setattr__(self, "state", normalized_state.copy())

        player_to_move = int(self.player_to_move)
        if player_to_move not in {BLACK, WHITE}:
            raise ValueError(f"player_to_move must be BLACK({BLACK}) or WHITE({WHITE}), got {player_to_move}.")
        object.__setattr__(self, "player_to_move", player_to_move)

        best_action = int(self.best_action)
        if best_action < -1:
            raise ValueError("best_action must be -1 (terminal sentinel) or non-negative.")
        object.__setattr__(self, "best_action", best_action)

        searched_value = float(self.searched_value)
        if not np.isfinite(searched_value):
            raise ValueError("searched_value must be finite.")
        object.__setattr__(self, "searched_value", searched_value)

        normalized_action_values: dict[int, float] = {}
        for action, value in self.action_values.items():
            action_key = int(action)
            action_value = float(value)
            if action_key < 0:
                raise ValueError(f"action_values key must be non-negative, got {action_key}.")
            if not np.isfinite(action_value):
                raise ValueError("action_values must contain only finite values.")
            normalized_action_values[action_key] = action_value
        object.__setattr__(self, "action_values", normalized_action_values)

        normalized_pv = [int(action) for action in self.principal_variation]
        for action in normalized_pv:
            if action < 0:
                raise ValueError(f"principal_variation must contain non-negative actions, got {action}.")
        object.__setattr__(self, "principal_variation", normalized_pv)

        nodes = int(self.nodes)
        if nodes < 0:
            raise ValueError("nodes must be non-negative.")
        object.__setattr__(self, "nodes", nodes)

        depth_reached = int(self.depth_reached)
        if depth_reached < 0:
            raise ValueError("depth_reached must be non-negative.")
        object.__setattr__(self, "depth_reached", depth_reached)

        object.__setattr__(self, "forced_tactical", bool(self.forced_tactical))

        if self.final_outcome is None:
            return
        final_outcome = float(self.final_outcome)
        if not np.isfinite(final_outcome):
            raise ValueError("final_outcome must be finite when provided.")
        if not -1.0 <= final_outcome <= 1.0:
            raise ValueError(f"final_outcome must be in [-1, 1], got {final_outcome}.")
        object.__setattr__(self, "final_outcome", final_outcome)

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to a plain-serializable dictionary."""

        return {
            "state": self.state.tolist(),
            "player_to_move": self.player_to_move,
            "best_action": self.best_action,
            "searched_value": self.searched_value,
            "action_values": {str(action): value for action, value in self.action_values.items()},
            "principal_variation": list(self.principal_variation),
            "nodes": self.nodes,
            "depth_reached": self.depth_reached,
            "forced_tactical": self.forced_tactical,
            "final_outcome": self.final_outcome,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AthenanReplaySample":
        """Build a sample from a dictionary payload."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")

        raw_action_values = payload.get("action_values", {})
        if not isinstance(raw_action_values, Mapping):
            raise ValueError("action_values must be a mapping.")

        normalized_action_values = {
            int(action): float(value) for action, value in dict(raw_action_values).items()
        }
        return cls(
            state=np.asarray(payload["state"], dtype=np.float32),
            player_to_move=int(payload["player_to_move"]),
            best_action=int(payload["best_action"]),
            searched_value=float(payload["searched_value"]),
            action_values=normalized_action_values,
            principal_variation=[int(action) for action in payload.get("principal_variation", [])],
            nodes=int(payload["nodes"]),
            depth_reached=int(payload["depth_reached"]),
            forced_tactical=bool(payload.get("forced_tactical", False)),
            final_outcome=None if payload.get("final_outcome") is None else float(payload["final_outcome"]),
        )
