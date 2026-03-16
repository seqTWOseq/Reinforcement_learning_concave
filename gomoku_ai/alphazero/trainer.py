"""Simple single-process baseline trainer for AlphaZero Gomoku."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from gomoku_ai.alphazero.model import PolicyValueNet
from gomoku_ai.alphazero.mcts import MCTSConfig
from gomoku_ai.alphazero.self_play import SelfPlayConfig, SelfPlayGameGenerator
from gomoku_ai.alphazero.specs import GameRecord, GameStepSample
from gomoku_ai.alphazero.trainer_utils import (
    build_dataloader_from_buffer,
    compute_policy_loss,
    compute_value_loss,
    trim_samples,
)

TRAINER_CHECKPOINT_FORMAT_VERSION = 1
TRAINER_CHECKPOINT_TYPE = "alphazero_trainer"


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the simple single-process AlphaZero baseline trainer.

    Fixed baseline defaults:
    - trainer_style = "simple single-process baseline"
    - num_self_play_games_per_cycle = 20
    - max_buffer_samples = 2000
    - batch_size = 64
    - epochs_per_cycle = 3
    - optimizer = AdamW
    - learning_rate = 1e-3
    - weight_decay = 1e-4
    - checkpoint_dir = "checkpoints/alphazero"
    - device = "cpu"
    """

    trainer_style: str = "simple single-process baseline"
    optimizer_name: str = "AdamW"
    policy_loss_name: str = "soft cross entropy"
    value_loss_name: str = "MSE"
    num_self_play_games_per_cycle: int = 20
    max_buffer_samples: int = 2000
    batch_size: int = 64
    epochs_per_cycle: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_dir: str = "checkpoints/alphazero"
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate trainer hyperparameters and fixed baseline choices."""

        if self.trainer_style != "simple single-process baseline":
            raise ValueError("trainer_style must be 'simple single-process baseline'.")
        if self.optimizer_name != "AdamW":
            raise ValueError("optimizer_name must be 'AdamW'.")
        if self.policy_loss_name != "soft cross entropy":
            raise ValueError("policy_loss_name must be 'soft cross entropy'.")
        if self.value_loss_name != "MSE":
            raise ValueError("value_loss_name must be 'MSE'.")
        if self.num_self_play_games_per_cycle <= 0:
            raise ValueError("num_self_play_games_per_cycle must be positive.")
        if self.max_buffer_samples <= 0:
            raise ValueError("max_buffer_samples must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs_per_cycle <= 0:
            raise ValueError("epochs_per_cycle must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if not self.checkpoint_dir or not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir must be a non-empty string.")
        if not self.device or not self.device.strip():
            raise ValueError("device must be a non-empty string.")


class RecentGameBuffer:
    """Recent-sample buffer that keeps only the latest N samples.

    The trimming policy is sample-based, not game-based. This trainer keeps the
    most recent `max_buffer_samples` individual `GameStepSample` entries.
    """

    def __init__(self, max_buffer_samples: int = 2000) -> None:
        """Initialize an empty recent-sample buffer."""

        if max_buffer_samples <= 0:
            raise ValueError("max_buffer_samples must be positive.")
        self.max_buffer_samples = max_buffer_samples
        self._samples: list[GameStepSample] = []

    def __len__(self) -> int:
        """Return the current number of buffered samples."""

        return len(self._samples)

    def add_game(self, record: GameRecord) -> None:
        """Append one game's samples and trim to the recent sample budget."""

        if not isinstance(record, GameRecord):
            raise TypeError("record must be a GameRecord instance.")
        self._samples.extend(record.samples)
        self.trim()

    def add_games(self, records: Sequence[GameRecord]) -> None:
        """Append multiple game records and trim once at the end."""

        for record in records:
            if not isinstance(record, GameRecord):
                raise TypeError("records must contain only GameRecord instances.")
            self._samples.extend(record.samples)
        self.trim()

    def get_all_samples(self) -> list[GameStepSample]:
        """Return all buffered samples in temporal order."""

        return list(self._samples)

    def trim(self) -> None:
        """Trim the buffer to the most recent sample budget."""

        self._samples = trim_samples(self._samples, self.max_buffer_samples)


class AlphaZeroTrainer:
    """Simple single-process baseline trainer for AlphaZero Gomoku.

    Training loss:
    - policy loss: soft cross entropy on distribution targets
    - value loss: MSE on `(N, 1)` value targets
    - total loss: `policy_loss + value_loss`

    Training metrics:
    - `num_training_samples` is the current buffered sample-entry count used to
      build the dataset for this training call. It is counted once per buffered
      `GameStepSample` entry and is not multiplied by `epochs_per_cycle`.
    """

    def __init__(
        self,
        model: PolicyValueNet,
        trainer_config: TrainerConfig | None = None,
        self_play_config: SelfPlayConfig | None = None,
        mcts_config: MCTSConfig | None = None,
    ) -> None:
        """Initialize the trainer, buffer, self-play generator, and optimizer."""

        if not isinstance(model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet instance.")

        self.trainer_config = trainer_config or TrainerConfig()
        self.self_play_config = self_play_config or SelfPlayConfig()
        self.mcts_config = mcts_config or MCTSConfig()
        self.device = torch.device(self.trainer_config.device)
        self.model = model.to(self.device)
        self.self_play_generator = SelfPlayGameGenerator(
            config=self.self_play_config,
            mcts_config=self.mcts_config,
        )
        self.buffer = RecentGameBuffer(max_buffer_samples=self.trainer_config.max_buffer_samples)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.trainer_config.learning_rate,
            weight_decay=self.trainer_config.weight_decay,
        )
        self.last_collected_games: list[GameRecord] = []
        self.last_checkpoint_path: Path | None = None

    def collect_self_play_games(
        self,
        model: PolicyValueNet | None = None,
        num_games: int | None = None,
    ) -> list[GameRecord]:
        """Collect one batch of self-play games with the model in eval mode."""

        active_model = self.model if model is None else model
        if not isinstance(active_model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet instance.")

        resolved_num_games = self.trainer_config.num_self_play_games_per_cycle if num_games is None else num_games
        if resolved_num_games <= 0:
            raise ValueError("num_games must be positive.")

        was_training = active_model.training
        active_model.eval()
        try:
            records = [self.self_play_generator.play_one_game(active_model) for _ in range(resolved_num_games)]
        finally:
            if was_training:
                active_model.train()

        self.last_collected_games = records
        return records

    def train_on_buffer(self) -> dict[str, float]:
        """Train the policy/value network on the current recent-sample buffer.

        Returned metrics:
        - `policy_loss`: average soft cross entropy across all processed batches
        - `value_loss`: average MSE across all processed batches
        - `total_loss`: average of `policy_loss + value_loss`
        - `num_training_samples`: current buffered sample-entry count used to
          build the dataset once before epoch repetition
        - `epochs`: configured number of epochs for this training call
        """

        samples = self.buffer.get_all_samples()
        if not samples:
            raise ValueError("Cannot train on an empty buffer.")

        dataloader = build_dataloader_from_buffer(
            samples,
            batch_size=self.trainer_config.batch_size,
            shuffle=True,
        )

        was_training = self.model.training
        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        total_policy_correct = 0
        total_value_outcome_correct = 0
        total_value_abs_error = 0.0
        total_samples = 0

        try:
            for _ in range(self.trainer_config.epochs_per_cycle):
                for state, policy_target, value_target in dataloader:
                    state = state.to(self.device, dtype=torch.float32)
                    policy_target = policy_target.to(self.device, dtype=torch.float32)
                    value_target = value_target.to(self.device, dtype=torch.float32)

                    policy_logits, value_pred = self.model(state)

                    # Policy loss uses soft cross entropy over distribution targets.
                    policy_loss = compute_policy_loss(policy_logits, policy_target)
                    # Value loss uses MSE on `(N, 1)` targets.
                    value_loss = compute_value_loss(value_pred, value_target)
                    batch_total_loss = policy_loss + value_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    batch_total_loss.backward()
                    self.optimizer.step()

                    batch_size = state.shape[0]
                    predicted_actions = policy_logits.argmax(dim=1)
                    target_actions = policy_target.argmax(dim=1)
                    total_policy_correct += int((predicted_actions == target_actions).sum().item())

                    value_pred_flat = value_pred.squeeze(1)
                    value_target_flat = value_target.squeeze(1)
                    predicted_outcome = torch.where(
                        value_pred_flat > 0.33,
                        torch.ones_like(value_pred_flat),
                        torch.where(
                            value_pred_flat < -0.33,
                            -torch.ones_like(value_pred_flat),
                            torch.zeros_like(value_pred_flat),
                        ),
                    )
                    target_outcome = torch.where(
                        value_target_flat > 0.0,
                        torch.ones_like(value_target_flat),
                        torch.where(
                            value_target_flat < 0.0,
                            -torch.ones_like(value_target_flat),
                            torch.zeros_like(value_target_flat),
                        ),
                    )
                    total_value_outcome_correct += int((predicted_outcome == target_outcome).sum().item())
                    total_value_abs_error += float(torch.abs(value_pred_flat - value_target_flat).sum().item())
                    total_samples += batch_size
                    total_policy_loss += float(policy_loss.item()) * batch_size
                    total_value_loss += float(value_loss.item()) * batch_size
                    total_loss += float(batch_total_loss.item()) * batch_size
        finally:
            if not was_training:
                self.model.eval()

        if total_samples <= 0:
            raise RuntimeError("Training loop completed without consuming any samples.")

        return {
            "policy_loss": total_policy_loss / total_samples,
            "value_loss": total_value_loss / total_samples,
            "total_loss": total_loss / total_samples,
            "policy_top1_accuracy": total_policy_correct / total_samples,
            "value_outcome_accuracy": total_value_outcome_correct / total_samples,
            "value_mae": total_value_abs_error / total_samples,
            # This is the current buffered sample-entry count, not an
            # epoch-multiplied processed-example total.
            "num_training_samples": float(len(samples)),
            "epochs": float(self.trainer_config.epochs_per_cycle),
        }

    def run_training_cycle(self, cycle_index: int) -> dict[str, float | str]:
        """Run one full baseline cycle: self-play, buffer update, training, checkpoint."""

        if cycle_index < 0:
            raise ValueError("cycle_index must be non-negative.")

        records = self.collect_self_play_games(num_games=self.trainer_config.num_self_play_games_per_cycle)
        self.buffer.add_games(records)
        metrics = self.train_on_buffer()
        checkpoint_path = self.save_checkpoint(cycle_index, metrics=metrics)

        enriched_metrics = dict(metrics)
        enriched_metrics["num_self_play_games"] = float(len(records))
        enriched_metrics["buffer_samples"] = float(len(self.buffer))
        enriched_metrics["checkpoint_path"] = str(checkpoint_path)
        return enriched_metrics

    def save_checkpoint(
        self,
        path_or_cycle_index: int | str | Path,
        *,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save a trainer checkpoint with model, optimizer, config, and metrics.

        Trainer checkpoints use their own metadata fields so they do not get
        confused with standalone model checkpoints.
        """

        if isinstance(path_or_cycle_index, int):
            if path_or_cycle_index < 0:
                raise ValueError("cycle_index must be non-negative.")
            checkpoint_dir = Path(self.trainer_config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"trainer_cycle_{path_or_cycle_index:04d}.pt"
            cycle_index = path_or_cycle_index
        else:
            checkpoint_path = Path(path_or_cycle_index)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            cycle_index = -1

        payload: dict[str, Any] = {
            "format_version": TRAINER_CHECKPOINT_FORMAT_VERSION,
            "checkpoint_type": TRAINER_CHECKPOINT_TYPE,
            "model_state_dict": self.model.state_dict(),
            "model_config": asdict(self.model.config),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_config": asdict(self.trainer_config),
            "cycle_index": cycle_index,
            "metrics": dict(metrics or {}),
        }
        torch.save(payload, checkpoint_path)
        self.last_checkpoint_path = checkpoint_path
        return checkpoint_path
