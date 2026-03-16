"""Helpers for loading AlphaZero models from different checkpoint formats."""

from __future__ import annotations

from pathlib import Path

import torch

from gomoku_ai.alphazero.checkpoint_manager import BEST_MODEL_CHECKPOINT_TYPE
from gomoku_ai.alphazero.model import MODEL_TYPE, PolicyValueNet, PolicyValueNetConfig
from gomoku_ai.alphazero.trainer import TRAINER_CHECKPOINT_TYPE


def resolve_model_checkpoint_path(path_or_dir: str | Path) -> Path:
    """Resolve a usable model checkpoint path from a file or checkpoint directory."""

    resolved_path = Path(path_or_dir)
    if resolved_path.is_dir():
        best_model_path = resolved_path / "best_model.pt"
        if best_model_path.exists():
            return best_model_path

        trainer_checkpoints = sorted(resolved_path.glob("trainer_cycle_*.pt"))
        if trainer_checkpoints:
            return trainer_checkpoints[-1]

        model_checkpoints = sorted(resolved_path.glob("*.pt"))
        if model_checkpoints:
            return model_checkpoints[-1]

        raise FileNotFoundError(f"No checkpoint files were found in {resolved_path}.")

    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved_path}.")
    return resolved_path


def load_model_for_inference(
    path_or_dir: str | Path,
    device: torch.device | str | None = None,
) -> PolicyValueNet:
    """Load a policy/value network from model, best-model, or trainer checkpoints."""

    checkpoint_path = resolve_model_checkpoint_path(path_or_dir)
    map_location = None if device is None else torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if checkpoint.get("checkpoint_type") == BEST_MODEL_CHECKPOINT_TYPE:
        model = PolicyValueNet(config=PolicyValueNetConfig(**dict(checkpoint["model_config"])))
        model.load_state_dict(checkpoint["model_state_dict"])
    elif checkpoint.get("checkpoint_type") == TRAINER_CHECKPOINT_TYPE:
        model_config = checkpoint.get("model_config")
        if model_config is None:
            model = PolicyValueNet()
        else:
            model = PolicyValueNet(config=PolicyValueNetConfig(**dict(model_config)))
        model.load_state_dict(checkpoint["model_state_dict"])
    elif checkpoint.get("model_type") == MODEL_TYPE:
        model = PolicyValueNet(config=PolicyValueNetConfig(**dict(checkpoint["config"])))
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a standalone model checkpoint, "
            "a best-model checkpoint, or a trainer checkpoint."
        )

    if device is not None:
        model = model.to(torch.device(device))
    model.eval()
    return model
