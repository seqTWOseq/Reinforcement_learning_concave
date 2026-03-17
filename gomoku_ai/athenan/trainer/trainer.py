"""Value-centered trainer for Athenan replay samples."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from gomoku_ai.athenan.network import AthenanValueNet, load_athenan_value_net, save_athenan_value_net
from gomoku_ai.athenan.replay import AthenanReplayBuffer, AthenanReplaySample
from gomoku_ai.athenan.trainer.losses import build_value_training_loss
from gomoku_ai.athenan.utils import ATHENAN_FEATURE_PLANES, set_seed


@dataclass(frozen=True)
class AthenanTrainBatch:
    """Tensorized replay batch used by value training."""

    states: torch.Tensor  # shape: (N, 3, H, W), dtype: float32
    final_outcomes: torch.Tensor  # shape: (N, 1), dtype: float32
    searched_values: torch.Tensor  # shape: (N, 1), dtype: float32


def replay_samples_to_batch_tensors(
    samples: Sequence[AthenanReplaySample],
    *,
    device: torch.device | str | None = None,
) -> AthenanTrainBatch:
    """Convert replay samples to training tensors.

    Contract:
    - states: `(N, 3, H, W)` float32
    - final_outcomes: `(N, 1)` float32
    - searched_values: `(N, 1)` float32
    """

    if not samples:
        raise ValueError("samples must contain at least one replay sample.")

    first_state = np.asarray(samples[0].state, dtype=np.float32)
    if first_state.ndim != 3 or first_state.shape[0] != ATHENAN_FEATURE_PLANES:
        raise ValueError(f"Each sample.state must have shape (3, H, W), got {first_state.shape}.")
    expected_hw = first_state.shape[1:]

    states: list[np.ndarray] = []
    final_outcomes: list[float] = []
    searched_values: list[float] = []
    for index, sample in enumerate(samples):
        state = np.asarray(sample.state, dtype=np.float32)
        if state.shape != (ATHENAN_FEATURE_PLANES, expected_hw[0], expected_hw[1]):
            raise ValueError(
                f"All state shapes must match {(ATHENAN_FEATURE_PLANES, expected_hw[0], expected_hw[1])}, "
                f"but sample index {index} has {state.shape}."
            )
        if sample.final_outcome is None:
            raise ValueError(
                f"sample.final_outcome is None at index {index}. "
                "Trainer requires backfilled final_outcome targets."
            )
        states.append(state)
        final_outcomes.append(float(sample.final_outcome))
        searched_values.append(float(sample.searched_value))

    states_tensor = torch.from_numpy(np.stack(states, axis=0).astype(np.float32, copy=False)).to(torch.float32)
    final_tensor = torch.tensor(final_outcomes, dtype=torch.float32).unsqueeze(1)
    searched_tensor = torch.tensor(searched_values, dtype=torch.float32).unsqueeze(1)

    if device is not None:
        resolved_device = torch.device(device)
        states_tensor = states_tensor.to(resolved_device)
        final_tensor = final_tensor.to(resolved_device)
        searched_tensor = searched_tensor.to(resolved_device)

    return AthenanTrainBatch(
        states=states_tensor,
        final_outcomes=final_tensor,
        searched_values=searched_tensor,
    )


class AthenanTrainer:
    """Replay-buffer based value trainer for Athenan."""

    def __init__(
        self,
        *,
        model: AthenanValueNet | None = None,
        replay_buffer: AthenanReplayBuffer | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        aux_search_weight: float = 0.0,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if aux_search_weight < 0:
            raise ValueError("aux_search_weight must be non-negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.aux_search_weight = float(aux_search_weight)
        self.batch_size = int(batch_size)
        self.replay_buffer = replay_buffer if replay_buffer is not None else AthenanReplayBuffer()
        self.model = (model if model is not None else AthenanValueNet()).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self._rng = random.Random(seed)
        if seed is not None:
            set_seed(seed)
            torch.manual_seed(seed)

    def train_step(self, samples: Sequence[AthenanReplaySample]) -> dict[str, float | int]:
        """Run one optimizer step on one batch of replay samples."""

        batch = replay_samples_to_batch_tensors(samples, device=self.device)
        self.model.train()

        predictions = self.model(batch.states)
        loss_breakdown = build_value_training_loss(
            predictions,
            batch.final_outcomes,
            searched_values=batch.searched_values,
            aux_search_weight=self.aux_search_weight,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss_breakdown.total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(loss_breakdown.total_loss.detach().item()),
            "value_loss": float(loss_breakdown.value_loss.detach().item()),
            "aux_search_loss": float(loss_breakdown.aux_search_loss.detach().item()),
            "batch_size": int(batch.states.shape[0]),
        }

    def train_one_epoch(
        self,
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> dict[str, float | int]:
        """Train over the full replay buffer once."""

        samples = self.replay_buffer.samples()
        if not samples:
            raise ValueError("Replay buffer is empty.")

        resolved_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        indices = list(range(len(samples)))
        if shuffle:
            self._rng.shuffle(indices)

        weighted_total_loss = 0.0
        weighted_value_loss = 0.0
        weighted_aux_loss = 0.0
        seen = 0

        for start in range(0, len(indices), resolved_batch_size):
            batch_indices = indices[start : start + resolved_batch_size]
            batch_samples = [samples[index] for index in batch_indices]
            metrics = self.train_step(batch_samples)
            current_batch_size = int(metrics["batch_size"])
            weighted_total_loss += float(metrics["total_loss"]) * current_batch_size
            weighted_value_loss += float(metrics["value_loss"]) * current_batch_size
            weighted_aux_loss += float(metrics["aux_search_loss"]) * current_batch_size
            seen += current_batch_size

        return {
            "total_loss": weighted_total_loss / seen,
            "value_loss": weighted_value_loss / seen,
            "aux_search_loss": weighted_aux_loss / seen,
            "batch_size": resolved_batch_size,
        }

    def train_loop(
        self,
        *,
        num_epochs: int,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> list[dict[str, float | int]]:
        """Run simple multi-epoch training over replay buffer."""

        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        history: list[dict[str, float | int]] = []
        for _ in range(num_epochs):
            history.append(self.train_one_epoch(batch_size=batch_size, shuffle=shuffle))
        return history

    def run_training_cycle(
        self,
        cycle_index: int,
        *,
        num_epochs: int = 1,
        batch_size: int | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, float | int | str]:
        """Run one training cycle and optionally save a model checkpoint."""

        if cycle_index < 0:
            raise ValueError("cycle_index must be non-negative.")
        history = self.train_loop(num_epochs=num_epochs, batch_size=batch_size, shuffle=True)
        metrics = dict(history[-1])
        metrics["cycle_index"] = cycle_index
        metrics["num_epochs"] = num_epochs

        if checkpoint_path is not None:
            self.save_model_checkpoint(
                checkpoint_path,
                metadata={
                    "cycle_index": cycle_index,
                    "num_epochs": num_epochs,
                    "train_metrics": metrics,
                },
            )
            metrics["checkpoint_path"] = str(Path(checkpoint_path))
        return metrics

    def save_model_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Save model-only checkpoint via `save_athenan_value_net`."""

        trainer_metadata = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "aux_search_weight": self.aux_search_weight,
            "batch_size": self.batch_size,
        }
        if metadata is not None:
            trainer_metadata.update(dict(metadata))
        save_athenan_value_net(self.model, path, metadata=trainer_metadata)

    def load_model_checkpoint(self, path: str | Path) -> None:
        """Load model-only checkpoint and reset optimizer on loaded parameters."""

        loaded_model = load_athenan_value_net(path, device=self.device)
        self.model = loaded_model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
