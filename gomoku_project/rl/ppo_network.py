from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from gomoku_project.core.constants import BLACK, BOARD_SIZE, EMPTY, WHITE


def encode_observations(observations: np.ndarray) -> np.ndarray:
    boards = np.asarray(observations, dtype=np.int8)
    if boards.ndim == 2:
        boards = boards[None, ...]
    if boards.ndim != 3:
        raise ValueError("Expected observations with shape (H, W) or (B, H, W).")

    return np.stack(
        [
            boards == BLACK,
            boards == WHITE,
            boards == EMPTY,
        ],
        axis=1,
    ).astype(np.float32)


def observations_to_tensor(observations: np.ndarray, device: str | torch.device) -> torch.Tensor:
    encoded = encode_observations(observations)
    return torch.as_tensor(encoded, dtype=torch.float32, device=device)


def valid_actions_to_mask(valid_actions: Sequence[int], action_space_size: int) -> np.ndarray:
    mask = np.zeros(action_space_size, dtype=bool)
    mask[np.asarray(list(valid_actions), dtype=np.int64)] = True
    return mask


def normalize_policy_prior_scores(
    prior_scores: np.ndarray,
    valid_actions: Sequence[int],
    *,
    clip_value: float = 3.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    prior_array = np.asarray(prior_scores, dtype=np.float32)
    normalized = np.zeros_like(prior_array, dtype=np.float32)
    valid_indices = np.asarray(list(valid_actions), dtype=np.int64)
    if valid_indices.size == 0:
        return normalized

    valid_values = prior_array[valid_indices]
    if valid_values.size <= 1 or float(np.max(valid_values) - np.min(valid_values)) < epsilon:
        return normalized

    mean = float(valid_values.mean())
    std = float(valid_values.std())
    if std >= epsilon:
        scaled_values = (valid_values - mean) / std
    else:
        centered = valid_values - mean
        max_abs = float(np.max(np.abs(centered)))
        if max_abs < epsilon:
            return normalized
        scaled_values = centered / max_abs

    if clip_value > 0.0:
        scaled_values = np.clip(scaled_values, -clip_value, clip_value)
    normalized[valid_indices] = scaled_values.astype(np.float32)
    return normalized


def mask_policy_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.bool()
    invalid_fill = torch.full_like(logits, -1e9)
    return torch.where(action_mask, logits, invalid_fill)


def mix_policy_logits_with_prior(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    prior_scores: np.ndarray | torch.Tensor | None = None,
    beta: float | np.ndarray | torch.Tensor = 0.0,
) -> torch.Tensor:
    if prior_scores is None:
        return mask_policy_logits(logits, action_mask)

    prior_tensor = torch.as_tensor(prior_scores, dtype=logits.dtype, device=logits.device)
    if prior_tensor.ndim == 1:
        prior_tensor = prior_tensor.unsqueeze(0)
    if prior_tensor.shape != logits.shape:
        raise ValueError(
            f"Expected prior_scores shape {tuple(logits.shape)}, got {tuple(prior_tensor.shape)}."
        )

    beta_tensor = torch.as_tensor(beta, dtype=logits.dtype, device=logits.device)
    if beta_tensor.ndim == 0:
        beta_tensor = beta_tensor.reshape(1, 1)
    elif beta_tensor.ndim == 1:
        beta_tensor = beta_tensor.unsqueeze(-1)

    mixed_logits = logits + beta_tensor * prior_tensor
    return mask_policy_logits(mixed_logits, action_mask)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class PPOActorCritic(nn.Module):
    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        hidden_channels: int = 64,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(*(ResidualBlock(hidden_channels) for _ in range(num_blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = torch.stack(
                [
                    (x == BLACK).float(),
                    (x == WHITE).float(),
                    (x == EMPTY).float(),
                ],
                dim=1,
            )
        elif x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected input with shape (H, W), (B, H, W), or (B, 3, H, W).")

        features = self.backbone(self.stem(x.float()))
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value
