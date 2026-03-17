"""Tests for Athenan self-play trajectory generation MVP."""

from __future__ import annotations

import numpy as np

from gomoku_ai.athenan.replay import AthenanReplayBuffer, winner_to_player_outcome
from gomoku_ai.athenan.search import AthenanSearcher
from gomoku_ai.athenan.trainer import AthenanSelfPlayRunner
from gomoku_ai.common.agents import BaseSearcher, SearchResult
from gomoku_ai.env import GomokuEnv


def _env_factory_5x5() -> GomokuEnv:
    return GomokuEnv(board_size=5)


class FirstLegalSearcher(BaseSearcher):
    """Fast deterministic test searcher that always picks first legal move."""

    def search(self, env: GomokuEnv) -> SearchResult:
        legal_actions = np.flatnonzero(np.asarray(env.get_valid_moves(), dtype=bool)).astype(int, copy=False)
        if legal_actions.size == 0:
            return SearchResult(
                best_action=-1,
                root_value=0.0,
                action_values={},
                principal_variation=[],
                nodes=1,
                depth_reached=0,
                forced_tactical=False,
            )
        action = int(legal_actions[0])
        return SearchResult(
            best_action=action,
            root_value=0.0,
            action_values={action: 0.0},
            principal_variation=[action],
            nodes=1,
            depth_reached=1,
            forced_tactical=False,
        )


class AlwaysZeroSearcher(BaseSearcher):
    """Searcher stub returning action 0, which becomes illegal after first move."""

    def search(self, env: GomokuEnv) -> SearchResult:
        return SearchResult(
            best_action=0,
            root_value=0.0,
            action_values={0: 0.0},
            principal_variation=[0],
            nodes=1,
            depth_reached=1,
            forced_tactical=False,
        )


class SentinelSearcher(BaseSearcher):
    """Searcher stub that always returns terminal sentinel action."""

    def search(self, env: GomokuEnv) -> SearchResult:
        return SearchResult(
            best_action=-1,
            root_value=0.0,
            action_values={},
            principal_variation=[],
            nodes=1,
            depth_reached=0,
            forced_tactical=False,
        )


def test_one_game_self_play_smoke() -> None:
    """One game should produce a valid summary and append samples to buffer."""

    buffer = AthenanReplayBuffer(max_size=1_000)
    runner = AthenanSelfPlayRunner(
        searcher=FirstLegalSearcher(),
        replay_buffer=buffer,
        env_factory=_env_factory_5x5,
        seed=2026,
    )

    summary = runner.play_one_game()

    assert summary.move_count > 0
    assert summary.move_count == len(summary.moves)
    assert summary.num_samples == summary.move_count
    assert summary.forced_tactical_count <= summary.move_count
    assert len(buffer) == summary.num_samples


def test_ten_game_batch_smoke() -> None:
    """Batch self-play should run multiple games and accumulate replay samples."""

    buffer = AthenanReplayBuffer(max_size=20_000)
    runner = AthenanSelfPlayRunner(
        searcher=FirstLegalSearcher(),
        replay_buffer=buffer,
        env_factory=_env_factory_5x5,
        seed=11,
    )

    summaries = runner.play_games(10)

    assert len(summaries) == 10
    assert sum(summary.num_samples for summary in summaries) == len(buffer)
    assert all(summary.move_count > 0 for summary in summaries)


def test_self_play_with_athenan_searcher_smoke() -> None:
    """Runner should interoperate with real `AthenanSearcher` implementation."""

    buffer = AthenanReplayBuffer(max_size=2_000)
    runner = AthenanSelfPlayRunner(
        searcher=AthenanSearcher(
            model=None,
            max_depth=1,
            candidate_limit=8,
            candidate_radius=1,
            use_alpha_beta=True,
        ),
        replay_buffer=buffer,
        env_factory=_env_factory_5x5,
        opening_random_steps=1,
        seed=123,
    )

    summary = runner.play_one_game()

    assert summary.move_count > 0
    assert summary.num_samples == summary.move_count
    assert len(buffer) == summary.num_samples


def test_self_play_moves_are_legal_and_non_repeating() -> None:
    """Self-play summary move list should not contain illegal duplicate moves."""

    buffer = AthenanReplayBuffer(max_size=1_000)
    runner = AthenanSelfPlayRunner(
        searcher=FirstLegalSearcher(),
        replay_buffer=buffer,
        env_factory=_env_factory_5x5,
    )

    summary = runner.play_one_game()
    board_area = 25

    assert len(summary.moves) == len(set(summary.moves))
    assert all(0 <= action < board_area for action in summary.moves)


def test_final_outcomes_are_backfilled_for_all_samples() -> None:
    """After game end, every stored sample should have final_outcome set."""

    buffer = AthenanReplayBuffer(max_size=1_000)
    runner = AthenanSelfPlayRunner(
        searcher=FirstLegalSearcher(),
        replay_buffer=buffer,
        env_factory=_env_factory_5x5,
    )

    summary = runner.play_one_game()
    samples = buffer.samples()

    assert samples
    assert all(sample.final_outcome is not None for sample in samples)
    assert all(
        sample.final_outcome == winner_to_player_outcome(
            winner=summary.winner,
            player_to_move=sample.player_to_move,
        )
        for sample in samples
    )


def test_opening_randomness_keeps_legal_moves_and_is_seed_reproducible() -> None:
    """Opening randomness should keep legal moves and be reproducible with same seed."""

    buffer_a = AthenanReplayBuffer(max_size=1_000)
    buffer_b = AthenanReplayBuffer(max_size=1_000)

    runner_a = AthenanSelfPlayRunner(
        searcher=FirstLegalSearcher(),
        replay_buffer=buffer_a,
        env_factory=_env_factory_5x5,
        opening_random_steps=3,
        seed=777,
    )
    runner_b = AthenanSelfPlayRunner(
        searcher=FirstLegalSearcher(),
        replay_buffer=buffer_b,
        env_factory=_env_factory_5x5,
        opening_random_steps=3,
        seed=777,
    )

    summary_a = runner_a.play_one_game()
    summary_b = runner_b.play_one_game()

    assert summary_a.moves[:3] == summary_b.moves[:3]
    assert len(summary_a.moves) == len(set(summary_a.moves))
    assert all(0 <= action < 25 for action in summary_a.moves)


def test_illegal_move_from_searcher_is_rejected() -> None:
    """Runner should raise on illegal selected action before apply_move."""

    runner = AthenanSelfPlayRunner(
        searcher=AlwaysZeroSearcher(),
        replay_buffer=AthenanReplayBuffer(max_size=1_000),
        env_factory=_env_factory_5x5,
    )

    try:
        runner.play_one_game()
    except RuntimeError as exc:
        assert "illegal" in str(exc).lower()
    else:
        raise AssertionError("Expected RuntimeError for illegal action.")


def test_terminal_sentinel_action_is_rejected() -> None:
    """Runner should fail fast when searcher returns best_action < 0 in loop."""

    runner = AthenanSelfPlayRunner(
        searcher=SentinelSearcher(),
        replay_buffer=AthenanReplayBuffer(max_size=1_000),
        env_factory=_env_factory_5x5,
    )

    try:
        runner.play_one_game()
    except RuntimeError as exc:
        assert "best_action < 0" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for sentinel best_action.")
