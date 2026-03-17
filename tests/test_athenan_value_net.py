"""Tests for the Athenan value-only network MVP."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from gomoku_ai.athenan.network import (
    AthenanValueNet,
    AthenanValueNetConfig,
    WarmStartReport,
    load_athenan_value_net,
    save_athenan_value_net,
    warm_start_value_net,
)
from gomoku_ai.athenan.utils import ATHENAN_FEATURE_SHAPE, encode_env_to_planes, env_to_tensor
from gomoku_ai.env import BLACK, BOARD_SIZE, GomokuEnv


def _temp_checkpoint_path() -> Path:
    return Path.cwd() / f"athenan_value_net_test_{uuid4().hex}.pt"


def test_forward_shape() -> None:
    """Forward pass should return `(N, 1)` values."""

    model = AthenanValueNet(AthenanValueNetConfig())
    x = torch.randn((4, 3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    value = model(x)

    assert value.shape == (4, 1)


def test_value_range_and_finite_output() -> None:
    """Value output should be finite and bounded by tanh."""

    model = AthenanValueNet()
    x = torch.randn((2, 3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    value = model(x)

    assert torch.isfinite(value).all()
    assert bool(torch.all(value <= 1.0 + 1e-6))
    assert bool(torch.all(value >= -1.0 - 1e-6))


def test_save_load_smoke() -> None:
    """Saved checkpoint should reconstruct an equivalent model output."""

    path = _temp_checkpoint_path()
    model = AthenanValueNet()
    model.eval()
    x = torch.randn((1, 3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)

    try:
        with torch.no_grad():
            before = model(x).clone()
        save_athenan_value_net(model, path)
        loaded = load_athenan_value_net(path, device="cpu")
        loaded.eval()
        with torch.no_grad():
            after = loaded(x).clone()
    finally:
        if path.exists():
            path.unlink()

    assert torch.allclose(before, after, atol=1e-6)


def test_board_encoder_shape() -> None:
    """Board encoder should produce fixed `(3, 15, 15)` float32 planes."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(env.coord_to_action(0, 0))
    env.apply_move(env.coord_to_action(1, 1))

    encoded = encode_env_to_planes(env)
    tensor = env_to_tensor(env)

    assert encoded.shape == ATHENAN_FEATURE_SHAPE
    assert encoded.dtype == np.float32
    assert encoded[0, 0, 0] == 1.0
    assert encoded[1, 1, 1] == 1.0
    assert np.all(encoded[2] == 1.0)
    assert tuple(tensor.shape) == (1, 3, BOARD_SIZE, BOARD_SIZE)
    assert tensor.dtype == torch.float32


def test_warm_start_scaffold_smoke() -> None:
    """Warm-start helper should report copied tensors and support cold start."""

    model = AthenanValueNet()
    state_dict = model.state_dict()
    target_key = next(iter(state_dict))
    source_state_dict = {target_key: torch.full_like(state_dict[target_key], 0.25)}

    report = warm_start_value_net(model, source_state_dict=source_state_dict)
    cold_report = warm_start_value_net(model, source_state_dict=None)

    assert isinstance(report, WarmStartReport)
    assert report.mode == "warm_start"
    assert report.copied_tensors == 1
    assert torch.allclose(model.state_dict()[target_key], source_state_dict[target_key])
    assert cold_report.mode == "cold_start"
    assert cold_report.copied_tensors == 0
