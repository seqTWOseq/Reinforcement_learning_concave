"""Board encoding helpers for Athenan value-only inference."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from gomoku_ai.env import BLACK, BOARD_SIZE, EMPTY, GomokuEnv

ATHENAN_FEATURE_PLANES = 3
ATHENAN_FEATURE_SHAPE = (ATHENAN_FEATURE_PLANES, BOARD_SIZE, BOARD_SIZE)
ATHENAN_FEATURE_DTYPE = np.float32


def encode_env_to_planes(env: GomokuEnv) -> np.ndarray:
    """Encode one `GomokuEnv` into Athenan value-network planes.

    Planes:
    0. current-player stones
    1. opponent stones
    2. current-player-to-move color plane (`1.0` for black-to-move, `0.0` for white-to-move)
    """

    current_stones = (env.board == env.current_player).astype(np.float32)
    opponent_stones = ((env.board != EMPTY) & (env.board != env.current_player)).astype(np.float32)
    to_move_plane = np.full(
        (env.board_size, env.board_size),
        1.0 if env.current_player == BLACK else 0.0,
        dtype=np.float32,
    )
    encoded = np.stack((current_stones, opponent_stones, to_move_plane), axis=0).astype(np.float32, copy=False)
    expected_shape = (ATHENAN_FEATURE_PLANES, env.board_size, env.board_size)
    if encoded.shape != expected_shape:
        raise ValueError(f"Encoded shape must be {expected_shape}, got {encoded.shape}.")
    return encoded


def env_to_tensor(env: GomokuEnv, device: torch.device | str | None = None) -> torch.Tensor:
    """Encode one environment and return an input tensor with shape `(1, C, H, W)`."""

    encoded = encode_env_to_planes(env)
    tensor = torch.from_numpy(encoded).unsqueeze(0).to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(torch.device(device))
    return tensor


def env_batch_to_tensor(
    envs: Sequence[GomokuEnv],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode multiple environments into a batch tensor `(N, C, H, W)`."""

    if not envs:
        raise ValueError("envs must contain at least one GomokuEnv instance.")
    stacked = np.stack([encode_env_to_planes(env) for env in envs], axis=0).astype(np.float32, copy=False)
    tensor = torch.from_numpy(stacked).to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(torch.device(device))
    return tensor
