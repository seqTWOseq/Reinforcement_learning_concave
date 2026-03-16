"""Tests for training dashboard generation and flexible checkpoint loading."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from gomoku_ai.alphazero import (
    AlphaZeroTrainer,
    GameRecord,
    GameStepSample,
    MCTSConfig,
    PolicyValueNet,
    PolicyValueNetConfig,
    SelfPlayConfig,
    TrainerConfig,
    append_training_log_entry,
    build_training_log_entry,
    load_model_for_inference,
    load_training_log_entries,
    resolve_model_checkpoint_path,
    summarize_self_play_records,
    write_training_dashboard_from_log,
)
from gomoku_ai.env import BLACK, BOARD_SIZE, DRAW, WHITE, GomokuEnv


def _make_sample(
    move_index: int,
    player_to_move: int,
    action_taken: int,
    value_target: float,
    *,
    game_id: str,
) -> GameStepSample:
    """Build one valid training sample for dashboard tests."""

    env = GomokuEnv()
    env.reset()
    state = env.encode_state()
    policy_target = np.zeros((BOARD_SIZE * BOARD_SIZE,), dtype=np.float32)
    policy_target[action_taken] = 1.0
    return GameStepSample(
        state=state,
        policy_target=policy_target,
        value_target=value_target,
        player_to_move=player_to_move,
        move_index=move_index,
        action_taken=action_taken,
        game_id=game_id,
    )


def _make_record(game_id: str, winner: int, moves: list[int]) -> GameRecord:
    """Build a compact synthetic self-play record."""

    samples = [
        _make_sample(
            index,
            BLACK if index % 2 == 0 else WHITE,
            action,
            1.0 if winner == (BLACK if index % 2 == 0 else WHITE) else -1.0 if winner in {BLACK, WHITE} else 0.0,
            game_id=game_id,
        )
        for index, action in enumerate(moves)
    ]
    return GameRecord(
        game_id=game_id,
        moves=moves,
        winner=winner,
        source="self_play",
        samples=samples,
        metadata={"num_moves": len(moves)},
    )


def test_summarize_self_play_records_returns_expected_rates() -> None:
    """Dashboard summaries should expose useful self-play win and length metrics."""

    records = [
        _make_record("game-black", BLACK, [0, 1, 2, 3, 4]),
        _make_record("game-white", WHITE, [10, 11, 12]),
        _make_record("game-draw", DRAW, [20, 21, 22, 23]),
    ]

    summary = summarize_self_play_records(records)

    assert summary["avg_game_length"] == 4.0
    assert summary["min_game_length"] == 3.0
    assert summary["max_game_length"] == 5.0
    assert summary["black_win_rate"] == 1.0 / 3.0
    assert summary["white_win_rate"] == 1.0 / 3.0
    assert summary["draw_rate"] == 1.0 / 3.0
    assert summary["avg_policy_entropy"] == 0.0


def test_training_dashboard_round_trip_creates_html() -> None:
    """JSONL log entries should round-trip into a static HTML dashboard."""

    workspace = Path.cwd() / f"dashboard_test_{uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "training_metrics.jsonl"
    html_path = workspace / "training_dashboard.html"
    record = _make_record("game-black", BLACK, [0, 1, 2, 3, 4])

    try:
        entry = build_training_log_entry(
            0,
            {
                "policy_loss": 1.0,
                "value_loss": 0.5,
                "total_loss": 1.5,
                "policy_top1_accuracy": 0.4,
                "value_outcome_accuracy": 0.75,
                "value_mae": 0.25,
                "num_training_samples": 5.0,
                "epochs": 1.0,
                "num_self_play_games": 1.0,
                "buffer_samples": 5.0,
                "checkpoint_path": "checkpoint.pt",
            },
            [record],
        )
        append_training_log_entry(log_path, entry)
        loaded_entries = load_training_log_entries(log_path)
        dashboard_path = write_training_dashboard_from_log(log_path, html_path)
        html = dashboard_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert len(loaded_entries) == 1
    assert dashboard_path == html_path
    assert "AlphaZero Training Dashboard" in html
    assert "Training Accuracy" in html
    assert "<svg" in html
    assert "Policy Top-1" in html


def test_load_model_for_inference_supports_trainer_checkpoints() -> None:
    """Interactive play should be able to load trainer checkpoints directly."""

    workspace = Path.cwd() / f"trainer_loader_test_{uuid4().hex}"
    checkpoint_dir = workspace / "checkpoints"
    trainer = AlphaZeroTrainer(
        model=PolicyValueNet(PolicyValueNetConfig(use_batch_norm=False)),
        trainer_config=TrainerConfig(
            num_self_play_games_per_cycle=1,
            max_buffer_samples=16,
            batch_size=4,
            epochs_per_cycle=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            checkpoint_dir=str(checkpoint_dir),
            device="cpu",
        ),
        self_play_config=SelfPlayConfig(
            opening_temperature_moves=0,
            opening_temperature=1.0,
            late_temperature=0.0,
            use_root_noise=False,
            game_id_prefix="selfplay",
        ),
        mcts_config=MCTSConfig(num_simulations=2, add_root_noise=False, temperature=0.0),
    )

    try:
        checkpoint_path = trainer.save_checkpoint(3, metrics={"total_loss": 1.23})
        payload = torch.load(checkpoint_path, map_location="cpu")
        resolved_path = resolve_model_checkpoint_path(checkpoint_dir)
        loaded_model = load_model_for_inference(checkpoint_dir, device="cpu")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert payload["checkpoint_type"] == "alphazero_trainer"
    assert "model_config" in payload
    assert resolved_path == checkpoint_path
    assert isinstance(loaded_model, PolicyValueNet)
    assert loaded_model.config.use_batch_norm is False
