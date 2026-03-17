"""Common agent/search interfaces used by multiple Gomoku pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from gomoku_ai.env import GomokuEnv


@dataclass(frozen=True)
class SearchResult:
    """Search output contract shared across engines.

    Notes:
    - `best_action` can be `-1` when the root position is terminal.
    - `nodes` counts visited search nodes, including the root node.
    """

    best_action: int
    root_value: float
    action_values: dict[int, float]
    principal_variation: list[int]
    nodes: int
    depth_reached: int
    forced_tactical: bool = False


class BaseSearcher(ABC):
    """Search component contract for one full root search call."""

    @abstractmethod
    def search(self, env: GomokuEnv) -> SearchResult:
        """Return a full `SearchResult` for the current position."""

    def select_action(self, env: GomokuEnv) -> int:
        """Compatibility helper that returns `search(env).best_action`."""

        return int(self.search(env).best_action)


class BaseAgent(ABC):
    """Agent contract that can play one move from a `GomokuEnv` state."""

    @abstractmethod
    def select_action(self, env: GomokuEnv) -> int:
        """Return one legal action for the current position."""
