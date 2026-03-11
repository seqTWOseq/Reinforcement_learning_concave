from __future__ import annotations

import torch
import torch.nn as nn

from gomoku_project.core.constants import BOARD_SIZE


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


class AlphaZeroPolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        hidden_channels: int = 32,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
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
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.float() / 2.0
        features = self.backbone(self.stem(x))
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


SimplePolicyNetwork = AlphaZeroPolicyValueNet
