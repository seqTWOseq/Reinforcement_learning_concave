"""Helpers that convert search outputs into replay samples."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from gomoku_ai.athenan.replay.schemas import AthenanReplaySample
from gomoku_ai.athenan.utils import encode_env_to_planes
from gomoku_ai.common.agents import SearchResult
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


def build_partial_replay_sample(
    env: GomokuEnv,
    search_result: SearchResult,
    *,
    selected_action: int | None = None,
) -> AthenanReplaySample:
    """Build one partial replay sample from the current root env + search output.

    Terminal sentinel guard:
    - `search_result.best_action < 0` is rejected because terminal states are not
      converted into replay samples.
    - `selected_action` can override search argmax for opening-randomized self-play.
    """

    if int(search_result.best_action) < 0:
        raise ValueError("search_result.best_action must be non-negative for partial replay samples.")

    resolved_action = int(search_result.best_action if selected_action is None else selected_action)
    if resolved_action < 0:
        raise ValueError("selected_action must be non-negative when provided.")

    principal_variation = [int(action) for action in search_result.principal_variation]
    if not principal_variation:
        principal_variation = [resolved_action]
    elif principal_variation[0] != resolved_action:
        principal_variation = [resolved_action] + principal_variation[1:]

    return AthenanReplaySample(
        state=encode_env_to_planes(env),
        player_to_move=int(env.current_player),
        best_action=resolved_action,
        searched_value=float(search_result.root_value),
        action_values={int(action): float(value) for action, value in search_result.action_values.items()},
        principal_variation=principal_variation,
        nodes=int(search_result.nodes),
        depth_reached=int(search_result.depth_reached),
        forced_tactical=bool(search_result.forced_tactical),
        final_outcome=None,
    )


def winner_to_player_outcome(*, winner: int, player_to_move: int) -> float:
    """Convert game winner to value from `player_to_move` perspective.

    Convention:
    - +1.0 if `player_to_move` eventually wins
    - -1.0 if `player_to_move` eventually loses
    - 0.0 on draw
    """

    normalized_winner = int(winner)
    normalized_player = int(player_to_move)

    if normalized_winner == DRAW:
        return 0.0
    if normalized_winner not in {BLACK, WHITE}:
        raise ValueError(f"winner must be BLACK({BLACK}), WHITE({WHITE}), or DRAW({DRAW}), got {winner}.")
    if normalized_player not in {BLACK, WHITE}:
        raise ValueError(
            f"player_to_move must be BLACK({BLACK}) or WHITE({WHITE}), got {player_to_move}."
        )
    return 1.0 if normalized_winner == normalized_player else -1.0


def backfill_final_outcomes(
    samples: Sequence[AthenanReplaySample],
    *,
    winner: int,
) -> list[AthenanReplaySample]:
    """Return new samples with `final_outcome` backfilled for one trajectory."""

    return [
        replace(
            sample,
            final_outcome=winner_to_player_outcome(winner=winner, player_to_move=sample.player_to_move),
        )
        for sample in samples
    ]
