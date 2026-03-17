"""Smoke tests for Athenan scaffolding and shared agent interfaces."""

from __future__ import annotations

import pytest

from gomoku_ai.alphazero import HumanPlayConfig, HumanVsAlphaZeroGameRunner
from gomoku_ai.athenan import AthenanDummyAgent
from gomoku_ai.env import BLACK, GomokuEnv, WHITE


def test_athenan_dummy_agent_returns_legal_move() -> None:
    """Dummy agent should always output one currently legal move."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(env.coord_to_action(0, 0))
    env.apply_move(env.coord_to_action(0, 1))

    agent = AthenanDummyAgent(pick_mode="first_legal")
    action = agent.select_action(env)

    assert action == env.coord_to_action(0, 2)
    assert bool(env.get_valid_moves()[action]) is True


def test_human_play_runner_accepts_shared_agent_interface() -> None:
    """The play entrypoint should accept interface-based external agents."""

    runner = HumanVsAlphaZeroGameRunner(
        config=HumanPlayConfig(
            human_color=BLACK,
            ai_temperature=0.0,
            use_root_noise=False,
            ai_num_simulations=1,
            record_ai_turn_only=True,
            game_id_prefix="humanplay",
        )
    )
    dummy_agent = AthenanDummyAgent(pick_mode="first_legal")
    helper_env = GomokuEnv()
    human_moves = [helper_env.coord_to_action(14, col) for col in range(5)]

    record = runner.play_game_with_agent(
        ai_agent=dummy_agent,
        human_color=BLACK,
        human_moves=human_moves,
    )

    assert record.source == "human_play"
    assert record.winner == BLACK
    assert len(record.samples) == 4
    assert all(sample.player_to_move == WHITE for sample in record.samples)
    for sample in record.samples:
        assert sample.policy_target[sample.action_taken] == pytest.approx(1.0)
        assert float(sample.policy_target.sum()) == pytest.approx(1.0)
