"""PyTorch value-only network baseline for Athenan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from gomoku_ai.athenan.utils.board_features import ATHENAN_FEATURE_PLANES, env_to_tensor
from gomoku_ai.env import BOARD_SIZE, GomokuEnv

ATHENAN_VALUE_NET_CHECKPOINT_FORMAT_VERSION = 1
ATHENAN_VALUE_NET_MODEL_TYPE = "athenan_value_net"


@dataclass(frozen=True)
class AthenanValueNetConfig:
    """Configuration for the Athenan value-only CNN baseline."""

    in_channels: int = ATHENAN_FEATURE_PLANES
    board_size: int = BOARD_SIZE
    trunk_channels: int = 64
    num_res_blocks: int = 3
    value_hidden_dim: int = 128
    use_batch_norm: bool = True

    def __post_init__(self) -> None:
        if self.in_channels != ATHENAN_FEATURE_PLANES:
            raise ValueError(f"in_channels must be {ATHENAN_FEATURE_PLANES}, got {self.in_channels}.")
        if self.board_size != BOARD_SIZE:
            raise ValueError(f"board_size must be {BOARD_SIZE}, got {self.board_size}.")
        if self.trunk_channels <= 0:
            raise ValueError("trunk_channels must be positive.")
        if self.num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive.")
        if self.value_hidden_dim <= 0:
            raise ValueError("value_hidden_dim must be positive.")


@dataclass(frozen=True)
class WarmStartReport:
    """Result summary for warm-start attempts."""

    mode: str
    copied_tensors: int
    total_target_tensors: int


class ConvBlock(nn.Module):
    """Conv-BN-ReLU block used in stem and residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_batch_norm,
            ),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Small residual block for board feature extraction."""

    def __init__(self, channels: int, *, use_batch_norm: bool) -> None:
        super().__init__()
        bias = not use_batch_norm
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.relu(out)


def _validate_input_tensor(x: torch.Tensor, config: AthenanValueNetConfig) -> None:
    """Validate forward input tensor contract `(N, C, H, W)`."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")
    if x.ndim != 4:
        raise ValueError(
            f"x must have shape (N, {config.in_channels}, {config.board_size}, {config.board_size}), got {tuple(x.shape)}."
        )
    if x.shape[0] <= 0:
        raise ValueError("Batch size must be positive.")
    if x.shape[1] != config.in_channels:
        raise ValueError(f"Channel dimension must be {config.in_channels}, got {x.shape[1]}.")
    if x.shape[2] != config.board_size or x.shape[3] != config.board_size:
        raise ValueError(
            f"Spatial dimensions must be ({config.board_size}, {config.board_size}), got {tuple(x.shape[2:])}."
        )
    if x.dtype != torch.float32:
        raise ValueError(f"x dtype must be torch.float32, got {x.dtype}.")
    if not torch.isfinite(x).all():
        raise ValueError("x must contain only finite values.")


class AthenanValueNet(nn.Module):
    """Value-only Gomoku network with tanh-bounded scalar output."""

    def __init__(self, config: AthenanValueNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or AthenanValueNetConfig()

        self.stem = ConvBlock(
            self.config.in_channels,
            self.config.trunk_channels,
            kernel_size=3,
            padding=1,
            use_batch_norm=self.config.use_batch_norm,
        )
        self.trunk = nn.Sequential(
            *[
                ResidualBlock(self.config.trunk_channels, use_batch_norm=self.config.use_batch_norm)
                for _ in range(self.config.num_res_blocks)
            ]
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(self.config.trunk_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.board_size * self.config.board_size, self.config.value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar state values with shape `(N, 1)` in `[-1, 1]`."""

        _validate_input_tensor(x, self.config)
        features = self.stem(x)
        features = self.trunk(features)
        return self.value_head(features)


def predict_env_value(
    model: AthenanValueNet,
    env: GomokuEnv,
    *,
    device: torch.device | str | None = None,
) -> float:
    """Convenience inference helper for occasional one-off env evaluation.

    Note:
    - this helper does not move model parameters between devices
    - caller should place the model once (for example, `model.to(device)`)
    """

    model_device = _infer_model_device(model)
    if device is not None:
        requested_device = torch.device(device)
        if requested_device != model_device:
            raise ValueError(
                f"Requested device {requested_device} does not match model device {model_device}. "
                "Move the model once before calling predict_env_value."
            )
    resolved_device = model_device
    tensor = env_to_tensor(env, device=resolved_device)
    was_training = model.training
    if was_training:
        model.eval()
    with torch.no_grad():
        value = model(tensor).reshape(-1)[0].item()
    if was_training:
        model.train()
    return float(value)


def count_parameters(model: nn.Module) -> int:
    """Return total parameter count."""

    return sum(parameter.numel() for parameter in model.parameters())


def save_athenan_value_net(
    model: AthenanValueNet,
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save model state/config to disk."""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": ATHENAN_VALUE_NET_CHECKPOINT_FORMAT_VERSION,
            "model_type": ATHENAN_VALUE_NET_MODEL_TYPE,
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "metadata": {} if metadata is None else dict(metadata),
        },
        resolved_path,
    )


def load_athenan_value_net(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> AthenanValueNet:
    """Load an Athenan value-net checkpoint."""

    map_location = None if device is None else torch.device(device)
    checkpoint = torch.load(Path(path), map_location=map_location)
    required_fields = {"format_version", "model_type", "state_dict", "config"}
    if not required_fields.issubset(checkpoint):
        raise ValueError(
            "Checkpoint must contain 'format_version', 'model_type', 'state_dict', and 'config'."
        )
    if checkpoint["format_version"] != ATHENAN_VALUE_NET_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format_version {checkpoint['format_version']!r}; "
            f"expected {ATHENAN_VALUE_NET_CHECKPOINT_FORMAT_VERSION}."
        )
    if checkpoint["model_type"] != ATHENAN_VALUE_NET_MODEL_TYPE:
        raise ValueError(
            f"Unsupported model_type {checkpoint['model_type']!r}; expected {ATHENAN_VALUE_NET_MODEL_TYPE!r}."
        )

    config = AthenanValueNetConfig(**dict(checkpoint["config"]))
    model = AthenanValueNet(config=config)
    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model = model.to(torch.device(device))
    return model


def warm_start_value_net(
    model: AthenanValueNet,
    *,
    source_state_dict: dict[str, torch.Tensor] | None = None,
) -> WarmStartReport:
    """Best-effort partial warm start using key/shape-compatible tensors.

    When `source_state_dict` is `None`, this intentionally does nothing and
    reports cold start.
    """

    target_state = model.state_dict()
    total_target_tensors = len(target_state)

    if source_state_dict is None:
        return WarmStartReport(mode="cold_start", copied_tensors=0, total_target_tensors=total_target_tensors)

    merged_state = dict(target_state)
    copied_tensors = 0
    for key, source_tensor in source_state_dict.items():
        if key not in merged_state:
            continue
        target_tensor = merged_state[key]
        if target_tensor.shape != source_tensor.shape:
            continue
        merged_state[key] = source_tensor.detach().to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        ).clone()
        copied_tensors += 1

    model.load_state_dict(merged_state, strict=False)
    return WarmStartReport(
        mode="warm_start",
        copied_tensors=copied_tensors,
        total_target_tensors=total_target_tensors,
    )


def warm_start_from_alphazero_checkpoint(
    model: AthenanValueNet,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> WarmStartReport:
    """Scaffold hook for future AlphaZero trunk transfer.

    TODO: Add key remapping rules once the final shared trunk naming is fixed.
    """

    map_location = None if device is None else torch.device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        candidate_state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        candidate_state_dict = checkpoint
    else:
        raise ValueError("checkpoint_path must contain a dict-like checkpoint payload.")

    normalized_state_dict = {str(key): value for key, value in dict(candidate_state_dict).items()}
    return warm_start_value_net(model, source_state_dict=normalized_state_dict)


def _infer_model_device(model: nn.Module) -> torch.device:
    """Infer model device from the first parameter; default to CPU if absent."""

    try:
        first_param = next(model.parameters())
    except StopIteration:
        return torch.device("cpu")
    return first_param.device
