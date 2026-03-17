"""Tests for Athenan trainer/evaluator stage."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from gomoku_ai.athenan.eval import AthenanEvaluator
from gomoku_ai.athenan.replay import AthenanReplayBuffer, AthenanReplaySample
from gomoku_ai.athenan.trainer import AthenanTrainer, replay_samples_to_batch_tensors
from gomoku_ai.env import BLACK, GomokuEnv


def _sample(
    *,
    final_outcome: float | None,
    searched_value: float = 0.0,
    marker: tuple[int, int] = (7, 7),
) -> AthenanReplaySample:
    state = np.zeros((3, 15, 15), dtype=np.float32)
    state[0, marker[0], marker[1]] = 1.0
    return AthenanReplaySample(
        state=state,
        player_to_move=BLACK,
        best_action=marker[0] * 15 + marker[1],
        searched_value=searched_value,
        action_values={marker[0] * 15 + marker[1]: searched_value},
        principal_variation=[marker[0] * 15 + marker[1]],
        nodes=8,
        depth_reached=2,
        forced_tactical=False,
        final_outcome=final_outcome,
    )


def test_replay_samples_to_batch_tensors_shape() -> None:
    """Batch conversion helper should return fixed training tensor shapes."""

    samples = [
        _sample(final_outcome=1.0, searched_value=0.2, marker=(7, 7)),
        _sample(final_outcome=-1.0, searched_value=-0.3, marker=(7, 8)),
    ]
    batch = replay_samples_to_batch_tensors(samples, device="cpu")

    assert tuple(batch.states.shape) == (2, 3, 15, 15)
    assert tuple(batch.final_outcomes.shape) == (2, 1)
    assert tuple(batch.searched_values.shape) == (2, 1)
    assert batch.states.dtype == torch.float32
    assert batch.final_outcomes.dtype == torch.float32
    assert batch.searched_values.dtype == torch.float32


def test_trainer_rejects_sample_without_final_outcome() -> None:
    """Trainer should fail fast when final_outcome target is missing."""

    buffer = AthenanReplayBuffer(max_size=16)
    buffer.add(_sample(final_outcome=None, searched_value=0.0))
    trainer = AthenanTrainer(replay_buffer=buffer, batch_size=1, device="cpu")

    try:
        trainer.train_one_epoch(shuffle=False)
    except ValueError as exc:
        assert "final_outcome" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing final_outcome.")


def test_trainer_keeps_passed_empty_replay_buffer_instance() -> None:
    """Passing an empty replay buffer should keep the same instance."""

    buffer = AthenanReplayBuffer(max_size=16)
    trainer = AthenanTrainer(replay_buffer=buffer, batch_size=1, device="cpu")

    assert trainer.replay_buffer is buffer


def test_trainer_one_step_smoke() -> None:
    """One optimizer step should run and return finite metrics."""

    trainer = AthenanTrainer(batch_size=2, aux_search_weight=0.2, device="cpu")
    metrics = trainer.train_step(
        [
            _sample(final_outcome=1.0, searched_value=0.6, marker=(7, 7)),
            _sample(final_outcome=-1.0, searched_value=-0.5, marker=(7, 8)),
        ]
    )

    assert metrics["batch_size"] == 2
    assert np.isfinite(metrics["total_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["aux_search_loss"])


def test_trainer_metrics_return_on_epoch() -> None:
    """One epoch on replay buffer should return required metric keys."""

    buffer = AthenanReplayBuffer(max_size=32)
    buffer.extend(
        [
            _sample(final_outcome=1.0, searched_value=0.4, marker=(7, 7)),
            _sample(final_outcome=1.0, searched_value=0.5, marker=(8, 7)),
            _sample(final_outcome=-1.0, searched_value=-0.2, marker=(7, 8)),
            _sample(final_outcome=-1.0, searched_value=-0.4, marker=(8, 8)),
        ]
    )
    trainer = AthenanTrainer(replay_buffer=buffer, batch_size=2, aux_search_weight=0.1, device="cpu")
    metrics = trainer.train_one_epoch(batch_size=2, shuffle=False)

    assert {"total_loss", "value_loss", "aux_search_loss", "batch_size"} == set(metrics.keys())
    assert metrics["batch_size"] == 2
    assert np.isfinite(metrics["total_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["aux_search_loss"])


def test_small_replay_buffer_overfit_smoke() -> None:
    """Repeated training on a tiny fixed batch should reduce value loss."""

    torch.manual_seed(0)
    trainer = AthenanTrainer(batch_size=8, learning_rate=1e-2, aux_search_weight=0.0, device="cpu", seed=0)
    fixed_batch = [_sample(final_outcome=1.0, searched_value=1.0, marker=(7, 7)) for _ in range(8)]

    initial_value_loss = float(trainer.train_step(fixed_batch)["value_loss"])
    final_value_loss = initial_value_loss
    for _ in range(25):
        final_value_loss = float(trainer.train_step(fixed_batch)["value_loss"])

    assert final_value_loss < initial_value_loss


def test_trainer_checkpoint_save_load_smoke() -> None:
    """Saved model checkpoint should restore identical forward outputs."""

    checkpoint_path = Path.cwd() / f"athenan_trainer_ckpt_{uuid4().hex}.pt"
    trainer = AthenanTrainer(device="cpu")
    x = torch.randn((2, 3, 15, 15), dtype=torch.float32)

    try:
        with torch.no_grad():
            before = trainer.model(x).clone()
        trainer.save_model_checkpoint(checkpoint_path)

        restored = AthenanTrainer(device="cpu")
        restored.load_model_checkpoint(checkpoint_path)
        with torch.no_grad():
            after = restored.model(x).clone()
    finally:
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    assert torch.allclose(before, after, atol=1e-6)


def test_evaluator_random_baseline_smoke() -> None:
    """Evaluator should produce finite W/L/D summary against random baseline."""

    evaluator = AthenanEvaluator(env_factory=lambda: GomokuEnv(board_size=5))
    summary = evaluator.evaluate_model_vs_random(
        model=None,
        num_games=2,
        random_seed=123,
        searcher_max_depth=1,
        searcher_candidate_limit=8,
        searcher_candidate_radius=1,
        searcher_use_alpha_beta=True,
    )

    assert summary.games == 2
    assert summary.wins + summary.losses + summary.draws == 2
    assert 0.0 <= summary.win_rate <= 1.0
    assert summary.avg_move_count > 0.0
