"""Tests for Athenan inference search + evaluator/human-play integration."""

from __future__ import annotations

import math

import numpy as np
import torch

from gomoku_ai.athenan.eval import AthenanEvaluator, RandomLegalAgent
from gomoku_ai.athenan.network import AthenanValueNet
from gomoku_ai.athenan.play import play_human_vs_athenan_game
from gomoku_ai.athenan.search import AthenanInferenceSearcher, AthenanSearcher
from gomoku_ai.common.agents import SearchResult
from gomoku_ai.env import BLACK, GomokuEnv, WHITE


def _place_fixture(env: GomokuEnv) -> None:
    env.reset()
    env.apply_move(env.coord_to_action(7, 7))
    env.apply_move(env.coord_to_action(7, 8))
    env.apply_move(env.coord_to_action(8, 7))
    env.apply_move(env.coord_to_action(8, 8))


def _place_immediate_block_fixture(env: GomokuEnv) -> int:
    env.reset()
    env.board[8, 2] = BLACK
    env.board[0, 0] = BLACK
    env.board[8, 3] = WHITE
    env.board[8, 4] = WHITE
    env.board[8, 5] = WHITE
    env.board[8, 6] = WHITE
    env.current_player = BLACK
    env.done = False
    env.winner = None
    env.last_move = None
    env.move_count = 6
    return env.coord_to_action(8, 7)


class ConstantValueNet(AthenanValueNet):
    """Model stub returning fixed non-zero value for tactical block tests."""

    def forward(self, x):  # type: ignore[override]
        return torch.full((x.shape[0], 1), 0.5, dtype=x.dtype, device=x.device)


def test_train_and_inference_search_both_return_legal_actions_same_model() -> None:
    """Both training and inference searchers should return legal actions."""

    env = GomokuEnv()
    _place_fixture(env)
    model = AthenanValueNet()

    train_result = AthenanSearcher(
        model=model,
        max_depth=1,
        candidate_limit=12,
        candidate_radius=1,
    ).search(env)
    infer_result = AthenanInferenceSearcher(
        model=model,
        max_depth=2,
        candidate_limit=16,
        candidate_radius=1,
        iterative_deepening=True,
        time_budget_sec=0.2,
    ).search(env)

    legal_mask = env.get_valid_moves()
    assert bool(legal_mask[train_result.best_action]) is True
    assert bool(legal_mask[infer_result.best_action]) is True


def test_inference_immediate_block_keeps_computed_root_value() -> None:
    """Inference search should block immediate loss and keep search-computed value."""

    env = GomokuEnv()
    block_action = _place_immediate_block_fixture(env)
    result = AthenanInferenceSearcher(
        model=ConstantValueNet(),
        max_depth=1,
        candidate_limit=8,
        candidate_radius=2,
        iterative_deepening=False,
    ).search(env)

    assert result.forced_tactical is True
    assert result.best_action == block_action
    assert set(result.action_values.keys()) == {block_action}
    assert abs(result.root_value) > 1e-6
    assert result.root_value == result.action_values[block_action]


def test_inference_search_returns_full_search_result_payload() -> None:
    """Inference search should satisfy SearchResult contract with populated fields."""

    env = GomokuEnv()
    _place_fixture(env)
    result = AthenanInferenceSearcher(
        model=None,
        max_depth=2,
        candidate_limit=14,
        candidate_radius=1,
        iterative_deepening=True,
    ).search(env)

    assert isinstance(result, SearchResult)
    assert result.action_values
    assert result.principal_variation
    assert result.principal_variation[0] == result.best_action
    assert math.isfinite(result.root_value)
    assert result.depth_reached >= 1
    assert result.nodes >= 1


def test_inference_search_depth_and_time_budget_options_are_reflected() -> None:
    """Depth and time-budget settings should be reflected in resulting depth."""

    env = GomokuEnv()
    _place_fixture(env)

    fixed_depth_result = AthenanInferenceSearcher(
        model=None,
        max_depth=1,
        candidate_limit=10,
        candidate_radius=1,
        iterative_deepening=False,
        time_budget_sec=None,
    ).search(env)
    untimed_result = AthenanInferenceSearcher(
        model=None,
        max_depth=3,
        candidate_limit=20,
        candidate_radius=2,
        iterative_deepening=True,
        time_budget_sec=None,
    ).search(env)
    timed_result = AthenanInferenceSearcher(
        model=None,
        max_depth=3,
        candidate_limit=20,
        candidate_radius=2,
        iterative_deepening=True,
        time_budget_sec=1e-9,
    ).search(env)

    assert fixed_depth_result.depth_reached == 1
    assert 1 <= timed_result.depth_reached <= 3
    assert timed_result.depth_reached <= untimed_result.depth_reached


def test_evaluator_inference_search_baseline_smoke() -> None:
    """Evaluator should run stronger inference search against random baseline."""

    evaluator = AthenanEvaluator(env_factory=lambda: GomokuEnv(board_size=5))
    summary = evaluator.evaluate_inference_search_vs_random(
        model=None,
        num_games=2,
        random_seed=123,
        searcher_max_depth=2,
        searcher_candidate_limit=10,
        searcher_candidate_radius=1,
        iterative_deepening=True,
        time_budget_sec=0.05,
    )

    assert summary.games == 2
    assert summary.wins + summary.losses + summary.draws == 2
    assert 0.0 <= summary.win_rate <= 1.0
    assert 0.0 <= summary.score_rate <= 1.0


def test_evaluator_can_compare_train_vs_inference_helpers() -> None:
    """Evaluator helper should return side-by-side train/inference summaries."""

    evaluator = AthenanEvaluator(env_factory=lambda: GomokuEnv(board_size=5))
    comparison = evaluator.evaluate_train_vs_inference(
        model=None,
        opponent_agent=RandomLegalAgent(seed=7),
        num_games=2,
        train_search_kwargs={
            "max_depth": 1,
            "candidate_limit": 8,
            "candidate_radius": 1,
        },
        inference_search_kwargs={
            "max_depth": 2,
            "candidate_limit": 12,
            "candidate_radius": 1,
            "iterative_deepening": True,
            "time_budget_sec": 0.05,
        },
    )

    assert comparison.train_summary.games == 2
    assert comparison.inference_summary.games == 2
    assert 0.0 <= comparison.train_summary.score_rate <= 1.0
    assert 0.0 <= comparison.inference_summary.score_rate <= 1.0


def test_human_play_entrypoint_function_smoke_with_debug() -> None:
    """Human-play function should complete game and return AI debug payloads."""

    searcher = AthenanInferenceSearcher(
        model=None,
        max_depth=2,
        candidate_limit=12,
        candidate_radius=1,
        iterative_deepening=True,
        time_budget_sec=0.05,
    )

    def first_legal_human_move(env: GomokuEnv) -> int:
        legal_actions = np.flatnonzero(np.asarray(env.get_valid_moves(), dtype=bool))
        return int(legal_actions[0])

    result = play_human_vs_athenan_game(
        searcher=searcher,
        human_color=BLACK,
        env_factory=lambda: GomokuEnv(board_size=5),
        human_move_selector=first_legal_human_move,
        debug=True,
        debug_top_k=3,
        print_fn=lambda _: None,
    )

    assert result.move_count == len(result.moves)
    assert result.move_count > 0
    assert result.ai_debug_turns
    for turn in result.ai_debug_turns:
        assert turn.principal_variation
        assert turn.top_actions
        assert math.isfinite(turn.root_value)
