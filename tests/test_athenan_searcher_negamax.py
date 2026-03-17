"""Tests for Athenan negamax searcher MVP."""

from __future__ import annotations

import torch

from gomoku_ai.athenan.network import AthenanValueNet
from gomoku_ai.athenan.search.searcher import AthenanSearcher
from gomoku_ai.common.agents import SearchResult
from gomoku_ai.env import BLACK, GomokuEnv, WHITE


class RaisingValueNet(AthenanValueNet):
    """Model stub that must not be called in terminal/tactical short-circuits."""

    def forward(self, x):  # type: ignore[override]
        raise AssertionError("Value net forward should not have been called.")


class ConstantValueNet(AthenanValueNet):
    """Model stub that returns a fixed non-zero value for sign-flow tests."""

    def forward(self, x):  # type: ignore[override]
        return torch.full((x.shape[0], 1), 0.5, dtype=x.dtype, device=x.device)


def _place_stones(
    env: GomokuEnv,
    black_coords: list[tuple[int, int]],
    white_coords: list[tuple[int, int]],
    *,
    current_player: int,
) -> None:
    env.reset()
    for row, col in black_coords:
        env.board[row, col] = BLACK
    for row, col in white_coords:
        env.board[row, col] = WHITE
    env.current_player = current_player
    env.last_move = None
    env.done = False
    env.winner = None
    env.move_count = len(black_coords) + len(white_coords)


def test_terminal_state_is_handled_before_network_eval() -> None:
    """Terminal root must bypass model inference and return immediate result."""

    env = GomokuEnv()
    env.reset()
    env.done = True
    env.winner = BLACK
    env.current_player = WHITE

    searcher = AthenanSearcher(model=RaisingValueNet(), max_depth=2)
    result = searcher.search(env)

    assert isinstance(result, SearchResult)
    assert result.root_value == -1.0
    assert result.nodes == 1
    assert result.depth_reached == 0


def test_forced_tactical_rule_precedes_negamax_search() -> None:
    """Forced tactical cases should short-circuit before recursive search."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )

    searcher = AthenanSearcher(model=RaisingValueNet(), max_depth=3, candidate_limit=8)
    result = searcher.search(env)

    assert result.forced_tactical is True
    assert result.best_action == env.coord_to_action(7, 7)


def test_immediate_block_restricts_root_candidates_but_keeps_negamax_value() -> None:
    """Immediate-block positions should still run search and keep computed root value."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(8, 2), (0, 0)],
        white_coords=[(8, 3), (8, 4), (8, 5), (8, 6)],
        current_player=BLACK,
    )
    block_action = env.coord_to_action(8, 7)

    searcher = AthenanSearcher(model=ConstantValueNet(), max_depth=1, candidate_limit=8)
    result = searcher.search(env)

    assert result.forced_tactical is True
    assert result.best_action == block_action
    assert set(result.action_values.keys()) == {block_action}
    assert abs(result.root_value) > 1e-6
    assert result.root_value == result.action_values[block_action]


def test_search_works_without_model_fallback_mode() -> None:
    """Searcher should work with `model=None` via neutral leaf fallback."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(7, 8)],
        current_player=BLACK,
    )

    searcher = AthenanSearcher(model=None, max_depth=2, candidate_limit=10)
    result = searcher.search(env)

    assert bool(env.get_valid_moves()[result.best_action]) is True
    assert isinstance(result.root_value, float)


def test_tiny_depth_one_search_smoke() -> None:
    """Depth-1 negamax should produce a legal action and root stats."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7), (8, 7)],
        white_coords=[(7, 8)],
        current_player=BLACK,
    )

    searcher = AthenanSearcher(model=AthenanValueNet(), max_depth=1, candidate_limit=6)
    result = searcher.search(env)

    assert bool(env.get_valid_moves()[result.best_action]) is True
    assert result.depth_reached >= 1
    assert result.nodes >= 1


def test_root_action_values_are_returned() -> None:
    """Root search must populate `action_values` in root-player perspective."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    searcher = AthenanSearcher(model=None, max_depth=2, candidate_limit=8)
    result = searcher.search(env)

    assert result.action_values
    assert result.best_action in result.action_values
    for action, value in result.action_values.items():
        assert bool(env.get_valid_moves()[action]) is True
        assert isinstance(value, float)


def test_principal_variation_is_not_empty() -> None:
    """Principal variation should include at least the chosen root action."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(6, 6)],
        white_coords=[(6, 7)],
        current_player=BLACK,
    )

    searcher = AthenanSearcher(model=None, max_depth=2, candidate_limit=8)
    result = searcher.search(env)

    assert result.principal_variation
    assert result.principal_variation[0] == result.best_action


def test_symmetric_like_board_returns_legal_move() -> None:
    """On near-symmetric boards, search result should still be legal."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(7, 8)],
        current_player=BLACK,
    )

    searcher = AthenanSearcher(model=AthenanValueNet(), max_depth=1, candidate_limit=12)
    result = searcher.search(env)

    assert bool(env.get_valid_moves()[result.best_action]) is True


def test_alpha_beta_on_off_both_work() -> None:
    """Alpha-beta toggle should produce legal outputs in both modes."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7), (7, 9)],
        white_coords=[(7, 8), (8, 8)],
        current_player=BLACK,
    )
    model = AthenanValueNet()

    result_with_ab = AthenanSearcher(
        model=model,
        max_depth=2,
        candidate_limit=10,
        use_alpha_beta=True,
    ).search(env)
    result_without_ab = AthenanSearcher(
        model=model,
        max_depth=2,
        candidate_limit=10,
        use_alpha_beta=False,
    ).search(env)

    assert bool(env.get_valid_moves()[result_with_ab.best_action]) is True
    assert bool(env.get_valid_moves()[result_without_ab.best_action]) is True
