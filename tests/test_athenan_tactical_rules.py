"""Tests for Athenan tactical rules and search contracts."""

from __future__ import annotations

from gomoku_ai.athenan.search import (
    apply_forced_tactical_rule,
    find_immediate_blocking_actions,
    find_immediate_winning_actions,
    generate_proximity_candidates,
    score_action,
)
from gomoku_ai.common.agents import SearchResult
from gomoku_ai.env import BLACK, GomokuEnv, WHITE


def _place_stones(
    env: GomokuEnv,
    black_coords: list[tuple[int, int]],
    white_coords: list[tuple[int, int]],
    *,
    current_player: int,
) -> None:
    """Build a deterministic board fixture by direct board mutation."""

    env.reset()
    for row, col in black_coords:
        env.board[row, col] = BLACK
    for row, col in white_coords:
        env.board[row, col] = WHITE
    env.current_player = current_player
    env.last_move = None
    env.winner = None
    env.done = False
    env.move_count = len(black_coords) + len(white_coords)


def test_find_immediate_winning_actions_detects_forced_win() -> None:
    """Current player should detect one-move tactical wins."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(7, 2), (0, 0)],
        current_player=BLACK,
    )

    winning_actions = find_immediate_winning_actions(env)

    assert winning_actions == [env.coord_to_action(7, 7)]


def test_find_immediate_winning_actions_open_four_returns_both_ends() -> None:
    """Open-four should expose two immediate winning endpoints."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(6, 7), (6, 8), (6, 9), (6, 10)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )

    winning_actions = find_immediate_winning_actions(env)
    expected = sorted([env.coord_to_action(6, 6), env.coord_to_action(6, 11)])

    assert winning_actions == expected


def test_find_immediate_blocking_actions_detects_forced_block() -> None:
    """Current player should detect blocking actions against opponent one-move wins."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(8, 2), (0, 0)],
        white_coords=[(8, 3), (8, 4), (8, 5), (8, 6)],
        current_player=BLACK,
    )

    blocking_actions = find_immediate_blocking_actions(env)

    assert blocking_actions == [env.coord_to_action(8, 7)]


def test_generate_proximity_candidates_respects_candidate_limit() -> None:
    """Candidate generation should clamp to `candidate_limit`."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(10, 10)],
        current_player=BLACK,
    )

    candidates = generate_proximity_candidates(env, radius=2, candidate_limit=5)
    stones = [(7, 7), (10, 10)]

    assert len(candidates) == 5
    for action in candidates:
        row, col = env.action_to_coord(action)
        assert bool(env.get_valid_moves()[action]) is True
        assert any(abs(row - stone_row) <= 2 and abs(col - stone_col) <= 2 for stone_row, stone_col in stones)


def test_generate_proximity_candidates_empty_board_returns_center() -> None:
    """Empty board candidate generation should start from center move."""

    env = GomokuEnv()
    env.reset()

    candidates = generate_proximity_candidates(env, radius=2, candidate_limit=10)
    center_action = env.coord_to_action(env.board_size // 2, env.board_size // 2)

    assert candidates == [center_action]


def test_score_action_returns_float_and_prioritizes_immediate_win() -> None:
    """Ordering score should be numeric and prefer tactical wins."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(7, 2), (0, 0)],
        current_player=BLACK,
    )
    win_action = env.coord_to_action(7, 7)
    quiet_action = env.coord_to_action(0, 1)

    win_score = score_action(env, win_action)
    quiet_score = score_action(env, quiet_action)

    assert isinstance(win_score, float)
    assert isinstance(quiet_score, float)
    assert win_score > quiet_score


def test_search_result_dataclass_can_be_created() -> None:
    """SearchResult should support direct contract-level construction."""

    result = SearchResult(
        best_action=42,
        root_value=0.25,
        action_values={42: 0.25},
        principal_variation=[42],
        nodes=17,
        depth_reached=3,
        forced_tactical=False,
    )

    assert result.best_action == 42
    assert result.root_value == 0.25
    assert result.action_values[42] == 0.25
    assert result.principal_variation == [42]
    assert result.nodes == 17
    assert result.depth_reached == 3
    assert result.forced_tactical is False


def test_forced_tactical_rule_wraps_output_as_search_result() -> None:
    """Forced tactical output should already satisfy `SearchResult` contract."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(7, 2), (1, 1)],
        current_player=BLACK,
    )

    result = apply_forced_tactical_rule(env, candidate_limit=8)

    assert isinstance(result, SearchResult)
    assert result is not None
    assert result.forced_tactical is True
    assert result.best_action == env.coord_to_action(7, 7)
    assert result.principal_variation == [env.coord_to_action(7, 7)]
