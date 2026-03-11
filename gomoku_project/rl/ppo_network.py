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


def mask_policy_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.bool()
    invalid_fill = torch.full_like(logits, -1e9)
    return torch.where(action_mask, logits, invalid_fill)


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
