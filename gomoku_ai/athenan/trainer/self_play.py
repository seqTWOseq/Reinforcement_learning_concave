"""Self-play loop MVP for Athenan trajectory generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from gomoku_ai.athenan.replay import (
    AthenanReplayBuffer,
    AthenanReplaySample,
    backfill_final_outcomes,
    build_partial_replay_sample,
)
from gomoku_ai.athenan.utils import set_seed
from gomoku_ai.common.agents import BaseSearcher, SearchResult
from gomoku_ai.env import GomokuEnv


@dataclass(frozen=True)
class AthenanSelfPlayGameSummary:
    """Summary for one completed self-play game."""

    winner: int
    move_count: int
    moves: list[int]
    num_samples: int
    forced_tactical_count: int

    def to_dict(self) -> dict[str, int | list[int]]:
        """Convert summary to a plain dictionary."""

        return {
            "winner": self.winner,
            "move_count": self.move_count,
            "moves": list(self.moves),
            "num_samples": self.num_samples,
            "forced_tactical_count": self.forced_tactical_count,
        }


class AthenanSelfPlayRunner:
    """Generate self-play trajectories and append them to replay buffer."""

    def __init__(
        self,
        *,
        searcher: BaseSearcher,
        replay_buffer: AthenanReplayBuffer,
        env_factory: Callable[[], GomokuEnv] | None = None,
        opening_random_steps: int = 0,
        seed: int | None = None,
    ) -> None:
        if opening_random_steps < 0:
            raise ValueError("opening_random_steps must be non-negative.")
        self.searcher = searcher
        self.replay_buffer = replay_buffer
        self.env_factory = env_factory or GomokuEnv
        self.opening_random_steps = opening_random_steps
        self.seed = seed
        self._rng = random.Random(seed)
        if seed is not None:
            set_seed(seed)

    def play_one_game(self) -> AthenanSelfPlayGameSummary:
        """Run one full self-play game and append backfilled samples to buffer."""

        env = self.env_factory()
        env.reset()

        trajectory: list[AthenanReplaySample] = []
        moves: list[int] = []
        forced_tactical_count = 0

        while not env.done:
            search_result = self.searcher.search(env)
            self._ensure_non_terminal_action(search_result)

            action = self._select_action(
                env,
                search_result=search_result,
                turn_index=len(moves),
            )
            self._assert_legal_action(env, action)

            # `best_action` in replay sample means the action actually executed.
            # During opening randomness this can differ from `search_result.best_action`.
            sample = build_partial_replay_sample(
                env,
                search_result,
                selected_action=action,
            )
            trajectory.append(sample)
            forced_tactical_count += int(bool(search_result.forced_tactical))

            env.apply_move(action)
            moves.append(action)

        if env.winner is None:
            raise RuntimeError("Self-play ended but winner is None.")

        finalized_trajectory = backfill_final_outcomes(trajectory, winner=int(env.winner))
        if any(sample.final_outcome is None for sample in finalized_trajectory):
            raise RuntimeError("All trajectory samples must have final_outcome after backfill.")

        self.replay_buffer.extend(finalized_trajectory)
        return AthenanSelfPlayGameSummary(
            winner=int(env.winner),
            move_count=int(env.move_count),
            moves=moves,
            num_samples=len(finalized_trajectory),
            forced_tactical_count=forced_tactical_count,
        )

    def play_games(self, num_games: int) -> list[AthenanSelfPlayGameSummary]:
        """Run multiple self-play games sequentially."""

        if num_games <= 0:
            raise ValueError("num_games must be positive.")
        return [self.play_one_game() for _ in range(num_games)]

    def _select_action(
        self,
        env: GomokuEnv,
        *,
        search_result: SearchResult,
        turn_index: int,
    ) -> int:
        """Pick action from search result with optional opening randomness."""

        if turn_index < self.opening_random_steps:
            legal_mask = np.asarray(env.get_valid_moves(), dtype=bool)
            legal_actions = np.flatnonzero(legal_mask).astype(int, copy=False)
            if legal_actions.size == 0:
                raise RuntimeError("No legal actions available on a non-terminal state.")

            search_candidates = [
                int(action)
                for action in search_result.action_values.keys()
                if 0 <= int(action) < legal_mask.size and bool(legal_mask[int(action)])
            ]
            candidate_pool = sorted(set(search_candidates))
            if not candidate_pool:
                candidate_pool = legal_actions.tolist()
            return int(self._rng.choice(candidate_pool))
        return int(search_result.best_action)

    @staticmethod
    def _assert_legal_action(env: GomokuEnv, action: int) -> None:
        """Fail fast when action is invalid for current env state."""

        legal_mask = np.asarray(env.get_valid_moves(), dtype=bool)
        if not (0 <= int(action) < legal_mask.size):
            raise RuntimeError(f"Selected action {action} is out of range.")
        if not bool(legal_mask[int(action)]):
            raise RuntimeError(f"Selected action {action} is illegal for current position.")

    @staticmethod
    def _ensure_non_terminal_action(search_result: SearchResult) -> None:
        """Reject terminal sentinel actions in non-terminal self-play loop."""

        if int(search_result.best_action) < 0:
            raise RuntimeError(
                "Searcher returned best_action < 0 during self-play. "
                "Terminal states must be excluded before sample creation."
            )


def run_one_self_play_game(
    *,
    searcher: BaseSearcher,
    replay_buffer: AthenanReplayBuffer,
    env_factory: Callable[[], GomokuEnv] | None = None,
    opening_random_steps: int = 0,
    seed: int | None = None,
) -> AthenanSelfPlayGameSummary:
    """Convenience helper to run one self-play game."""

    runner = AthenanSelfPlayRunner(
        searcher=searcher,
        replay_buffer=replay_buffer,
        env_factory=env_factory,
        opening_random_steps=opening_random_steps,
        seed=seed,
    )
    return runner.play_one_game()
