"""Replay buffer for Athenan search-derived samples."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from gomoku_ai.athenan.replay.sample_builder import backfill_final_outcomes, build_partial_replay_sample
from gomoku_ai.athenan.replay.schemas import AthenanReplaySample
from gomoku_ai.common.agents import SearchResult
from gomoku_ai.env import GomokuEnv


class AthenanReplayBuffer:
    """In-memory replay buffer storing `AthenanReplaySample` objects."""

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        self.max_size = max_size
        self._items: list[AthenanReplaySample] = []

    def add(self, item: AthenanReplaySample) -> None:
        """Append one item and evict oldest data when full."""

        self._items.append(item)
        if len(self._items) > self.max_size:
            self._items.pop(0)

    def add_from_search_result(self, env: GomokuEnv, search_result: SearchResult) -> AthenanReplaySample:
        """Build and append one partial replay sample from root search output."""

        sample = build_partial_replay_sample(env, search_result)
        self.add(sample)
        return sample

    def extend(self, items: Sequence[AthenanReplaySample]) -> None:
        """Append multiple samples while respecting max size."""

        for item in items:
            self.add(item)

    def backfill_final_outcomes(
        self,
        *,
        winner: int,
        start_index: int = 0,
        end_index: int | None = None,
    ) -> None:
        """Backfill final outcomes for one trajectory slice in the buffer."""

        resolved_end = len(self._items) if end_index is None else end_index
        if not (0 <= start_index <= resolved_end <= len(self._items)):
            raise ValueError(
                f"Invalid backfill range: start_index={start_index}, end_index={resolved_end}, len={len(self._items)}."
            )
        replaced = backfill_final_outcomes(self._items[start_index:resolved_end], winner=winner)
        self._items[start_index:resolved_end] = replaced

    def samples(self) -> list[AthenanReplaySample]:
        """Return a shallow copy of all stored samples."""

        return list(self._items)

    def clear(self) -> None:
        """Remove all stored samples."""

        self._items.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert the buffer into a dictionary payload."""

        return {
            "max_size": self.max_size,
            "items": [item.to_dict() for item in self._items],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AthenanReplayBuffer":
        """Restore a buffer from a dictionary payload."""

        buffer = cls(max_size=int(payload["max_size"]))
        buffer.extend([AthenanReplaySample.from_dict(item) for item in payload.get("items", [])])
        return buffer

    def __iter__(self) -> Iterator[AthenanReplaySample]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


# Backward-compatible alias kept to avoid breaking early-stage imports.
ReplayItem = AthenanReplaySample
