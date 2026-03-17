"""Tests for Athenan replay schema/builder/buffer integration."""

from __future__ import annotations

import numpy as np

from gomoku_ai.athenan.replay import (
    AthenanReplayBuffer,
    AthenanReplaySample,
    backfill_final_outcomes,
    build_partial_replay_sample,
)
from gomoku_ai.common.agents import SearchResult
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


def _make_sample(
    *,
    player_to_move: int,
    best_action: int,
    searched_value: float = 0.25,
    final_outcome: float | None = None,
) -> AthenanReplaySample:
    state = np.zeros((3, 15, 15), dtype=np.float32)
    state[2] = 1.0 if player_to_move == BLACK else 0.0
    return AthenanReplaySample(
        state=state,
        player_to_move=player_to_move,
        best_action=best_action,
        searched_value=searched_value,
        action_values={best_action: searched_value},
        principal_variation=[best_action],
        nodes=5,
        depth_reached=2,
        forced_tactical=False,
        final_outcome=final_outcome,
    )


def test_build_partial_replay_sample_smoke_and_field_preservation() -> None:
    """SearchResult fields should be preserved in replay sample conversion."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(env.coord_to_action(0, 0))
    env.apply_move(env.coord_to_action(1, 1))

    best_action = env.coord_to_action(7, 7)
    search_result = SearchResult(
        best_action=best_action,
        root_value=0.33,
        action_values={best_action: 0.33, env.coord_to_action(7, 8): 0.1},
        principal_variation=[best_action, env.coord_to_action(7, 8)],
        nodes=19,
        depth_reached=3,
        forced_tactical=True,
    )

    sample = build_partial_replay_sample(env, search_result)

    assert sample.state.shape == (3, env.board_size, env.board_size)
    assert sample.state.dtype == np.float32
    assert sample.player_to_move == env.current_player
    assert sample.best_action == search_result.best_action
    assert sample.searched_value == search_result.root_value
    assert sample.action_values == search_result.action_values
    assert sample.principal_variation == search_result.principal_variation
    assert sample.nodes == search_result.nodes
    assert sample.depth_reached == search_result.depth_reached
    assert sample.forced_tactical is True
    assert sample.final_outcome is None


def test_build_partial_replay_sample_rejects_terminal_sentinel_action() -> None:
    """Terminal sentinel best_action should not be converted into partial sample."""

    env = GomokuEnv()
    env.reset()
    search_result = SearchResult(
        best_action=-1,
        root_value=0.0,
        action_values={},
        principal_variation=[],
        nodes=1,
        depth_reached=0,
        forced_tactical=False,
    )

    try:
        build_partial_replay_sample(env, search_result)
    except ValueError as exc:
        assert "best_action" in str(exc)
    else:
        raise AssertionError("Expected ValueError for terminal sentinel action.")


def test_backfill_final_outcomes_uses_player_to_move_sign_convention() -> None:
    """Winner should be mapped to +1/-1 from each sample player's perspective."""

    samples = [
        _make_sample(player_to_move=BLACK, best_action=10),
        _make_sample(player_to_move=WHITE, best_action=11),
    ]

    filled = backfill_final_outcomes(samples, winner=BLACK)

    assert filled[0].final_outcome == 1.0
    assert filled[1].final_outcome == -1.0
    assert samples[0].final_outcome is None
    assert samples[1].final_outcome is None


def test_backfill_final_outcomes_draw_is_zero() -> None:
    """Draw winner label should map to zero outcome for all samples."""

    samples = [
        _make_sample(player_to_move=BLACK, best_action=3),
        _make_sample(player_to_move=WHITE, best_action=4),
    ]

    filled = backfill_final_outcomes(samples, winner=DRAW)

    assert filled[0].final_outcome == 0.0
    assert filled[1].final_outcome == 0.0


def test_sample_dict_round_trip() -> None:
    """Replay sample should support dict conversion round trip."""

    sample = _make_sample(player_to_move=BLACK, best_action=42, searched_value=-0.2, final_outcome=1.0)

    payload = sample.to_dict()
    restored = AthenanReplaySample.from_dict(payload)

    assert np.array_equal(restored.state, sample.state)
    assert restored.player_to_move == sample.player_to_move
    assert restored.best_action == sample.best_action
    assert restored.searched_value == sample.searched_value
    assert restored.action_values == sample.action_values
    assert restored.principal_variation == sample.principal_variation
    assert restored.nodes == sample.nodes
    assert restored.depth_reached == sample.depth_reached
    assert restored.forced_tactical == sample.forced_tactical
    assert restored.final_outcome == sample.final_outcome


def test_replay_buffer_add_len_and_max_size_eviction() -> None:
    """Buffer should evict oldest entries when max_size is exceeded."""

    buffer = AthenanReplayBuffer(max_size=2)
    buffer.add(_make_sample(player_to_move=BLACK, best_action=1))
    buffer.add(_make_sample(player_to_move=WHITE, best_action=2))
    buffer.add(_make_sample(player_to_move=BLACK, best_action=3))

    best_actions = [sample.best_action for sample in buffer.samples()]
    assert len(buffer) == 2
    assert best_actions == [2, 3]


def test_buffer_add_from_search_result_preserves_search_payload() -> None:
    """Buffer helper should keep best_action/action_values/pv from SearchResult."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(env.coord_to_action(4, 4))
    env.apply_move(env.coord_to_action(4, 5))

    best_action = env.coord_to_action(6, 6)
    search_result = SearchResult(
        best_action=best_action,
        root_value=0.5,
        action_values={best_action: 0.5, env.coord_to_action(6, 7): 0.2},
        principal_variation=[best_action, env.coord_to_action(6, 7)],
        nodes=7,
        depth_reached=2,
        forced_tactical=False,
    )

    buffer = AthenanReplayBuffer(max_size=8)
    sample = buffer.add_from_search_result(env, search_result)
    stored = buffer.samples()[0]

    assert len(buffer) == 1
    assert stored == sample
    assert stored.best_action == search_result.best_action
    assert stored.action_values == search_result.action_values
    assert stored.principal_variation == search_result.principal_variation


def test_buffer_dict_round_trip_and_slice_backfill() -> None:
    """Buffer payload conversion and in-buffer backfill should both work."""

    buffer = AthenanReplayBuffer(max_size=4)
    buffer.add(_make_sample(player_to_move=BLACK, best_action=1))
    buffer.add(_make_sample(player_to_move=WHITE, best_action=2))
    buffer.add(_make_sample(player_to_move=BLACK, best_action=3))

    buffer.backfill_final_outcomes(winner=WHITE, start_index=1)
    payload = buffer.to_dict()
    restored = AthenanReplayBuffer.from_dict(payload)

    assert len(restored) == 3
    outcomes = [sample.final_outcome for sample in restored.samples()]
    assert outcomes == [None, 1.0, -1.0]
