"""Forced tactical rules and candidate generation for Athenan search."""

from __future__ import annotations

import numpy as np

from gomoku_ai.athenan.search.move_ordering import order_actions
from gomoku_ai.common.agents import SearchResult
from gomoku_ai.env import BLACK, EMPTY, GomokuEnv, WHITE


def find_immediate_winning_actions(
    env: GomokuEnv,
    *,
    player: int | None = None,
    candidate_actions: list[int] | None = None,
) -> list[int]:
    """Return legal actions that win immediately for `player`."""

    if env.done:
        return []

    player_to_move = env.current_player if player is None else int(player)
    actions = _resolve_action_pool(env, candidate_actions)
    winning_actions: list[int] = []
    for action in actions:
        if _is_immediate_win_for_player(env, action, player_to_move):
            winning_actions.append(action)
    return sorted(winning_actions)


def find_immediate_blocking_actions(
    env: GomokuEnv,
    *,
    defender: int | None = None,
    candidate_actions: list[int] | None = None,
) -> list[int]:
    """Return legal actions that explicitly occupy opponent immediate-win points."""

    if env.done:
        return []

    defender_player = env.current_player if defender is None else int(defender)
    attacker = WHITE if defender_player == BLACK else BLACK
    opponent_immediate_wins = find_immediate_winning_actions(
        env,
        player=attacker,
        candidate_actions=candidate_actions,
    )
    return _blocking_actions_from_opponent_immediate_wins(opponent_immediate_wins)


def generate_proximity_candidates(
    env: GomokuEnv,
    *,
    radius: int = 2,
    candidate_limit: int | None = None,
) -> list[int]:
    """Generate legal candidates within `radius` of existing stones."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if candidate_limit is not None and candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive when provided.")

    legal_moves = np.asarray(env.get_valid_moves(), dtype=bool)
    legal_actions = np.flatnonzero(legal_moves).astype(int, copy=False)
    if legal_actions.size == 0:
        return []

    occupied_coords = np.argwhere(env.board != EMPTY)
    if occupied_coords.size == 0:
        center_action = env.coord_to_action(env.board_size // 2, env.board_size // 2)
        base_actions = [center_action] if legal_moves[center_action] else [int(legal_actions[0])]
        return base_actions[:candidate_limit] if candidate_limit is not None else base_actions

    candidate_set: set[int] = set()
    for row, col in occupied_coords:
        for delta_row in range(-radius, radius + 1):
            for delta_col in range(-radius, radius + 1):
                next_row = int(row + delta_row)
                next_col = int(col + delta_col)
                if not (0 <= next_row < env.board_size and 0 <= next_col < env.board_size):
                    continue
                action = env.coord_to_action(next_row, next_col)
                if legal_moves[action]:
                    candidate_set.add(action)

    if not candidate_set:
        ordered_all = order_actions(env, legal_actions.tolist(), candidate_limit=candidate_limit)
        return ordered_all

    ordered_candidates = order_actions(
        env,
        sorted(candidate_set),
        candidate_limit=candidate_limit,
    )
    return ordered_candidates


def apply_forced_tactical_rule(
    env: GomokuEnv,
    *,
    candidate_limit: int | None = None,
) -> SearchResult | None:
    """Apply forced tactical rules and return a wrapped `SearchResult`."""

    immediate_wins = find_immediate_winning_actions(env)
    if immediate_wins:
        ordered_wins = order_actions(env, immediate_wins, candidate_limit=candidate_limit)
        best_action = ordered_wins[0]
        return SearchResult(
            best_action=best_action,
            root_value=1.0,
            action_values={action: 1.0 for action in ordered_wins},
            principal_variation=[best_action],
            nodes=max(1, len(ordered_wins)),
            depth_reached=1,
            forced_tactical=True,
        )

    immediate_blocks = find_immediate_blocking_actions(env)
    if immediate_blocks:
        ordered_blocks = order_actions(env, immediate_blocks, candidate_limit=candidate_limit)
        best_action = ordered_blocks[0]
        return SearchResult(
            best_action=best_action,
            root_value=0.0,
            action_values={action: 0.0 for action in ordered_blocks},
            principal_variation=[best_action],
            nodes=max(1, len(ordered_blocks)),
            depth_reached=1,
            forced_tactical=True,
        )

    return None


def _resolve_action_pool(env: GomokuEnv, candidate_actions: list[int] | None) -> list[int]:
    """Resolve a legal action pool for tactical checks."""

    legal_moves = np.asarray(env.get_valid_moves(), dtype=bool)
    if candidate_actions is None:
        return np.flatnonzero(legal_moves).astype(int, copy=False).tolist()

    resolved: list[int] = []
    seen: set[int] = set()
    for action in candidate_actions:
        normalized = int(action)
        if normalized in seen:
            continue
        if 0 <= normalized < legal_moves.size and legal_moves[normalized]:
            resolved.append(normalized)
            seen.add(normalized)
    return resolved


def _blocking_actions_from_opponent_immediate_wins(opponent_wins: list[int]) -> list[int]:
    """Map opponent immediate wins to defender blocking actions.

    In Gomoku, immediate wins are blocked by occupying those exact points.
    """

    return sorted({int(action) for action in opponent_wins})


def _is_immediate_win_for_player(env: GomokuEnv, action: int, player: int) -> bool:
    """Return `True` when `player` wins by playing `action` now."""

    cloned = env.clone()
    cloned.current_player = int(player)
    cloned.done = False
    cloned.winner = None
    try:
        cloned.apply_move(action)
    except ValueError:
        return False
    return bool(cloned.done and cloned.winner == player)
