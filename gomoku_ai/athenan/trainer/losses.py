"""Loss helpers for Athenan value-only training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AthenanLossBreakdown:
    """Container for value-training losses."""

    total_loss: torch.Tensor
    value_loss: torch.Tensor
    aux_search_loss: torch.Tensor


def compute_value_loss(
    predicted_values: torch.Tensor,
    final_outcomes: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error between predicted value and final outcome target."""

    _validate_column_tensor(predicted_values, name="predicted_values")
    _validate_column_tensor(final_outcomes, name="final_outcomes")
    _validate_same_shape(predicted_values, final_outcomes, "predicted_values", "final_outcomes")
    return F.mse_loss(predicted_values, final_outcomes)


def compute_aux_search_loss(
    predicted_values: torch.Tensor,
    searched_values: torch.Tensor,
) -> torch.Tensor:
    """Optional MSE against search-provided root value targets."""

    _validate_column_tensor(predicted_values, name="predicted_values")
    _validate_column_tensor(searched_values, name="searched_values")
    _validate_same_shape(predicted_values, searched_values, "predicted_values", "searched_values")
    return F.mse_loss(predicted_values, searched_values)


def build_value_training_loss(
    predicted_values: torch.Tensor,
    final_outcomes: torch.Tensor,
    *,
    searched_values: torch.Tensor | None = None,
    aux_search_weight: float = 0.0,
) -> AthenanLossBreakdown:
    """Build total/value/aux loss tensors for one training batch."""

    if aux_search_weight < 0.0:
        raise ValueError("aux_search_weight must be non-negative.")

    value_loss = compute_value_loss(predicted_values, final_outcomes)
    if searched_values is None:
        aux_search_loss = torch.zeros((), dtype=value_loss.dtype, device=value_loss.device)
    else:
        aux_search_loss = compute_aux_search_loss(predicted_values, searched_values)

    total_loss = value_loss + (float(aux_search_weight) * aux_search_loss)
    return AthenanLossBreakdown(
        total_loss=total_loss,
        value_loss=value_loss,
        aux_search_loss=aux_search_loss,
    )


def _validate_column_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor.")
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape (N, 1), got {tuple(tensor.shape)}.")
    if tensor.shape[0] <= 0:
        raise ValueError(f"{name} batch dimension must be positive.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")


def _validate_same_shape(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_name: str,
    rhs_name: str,
) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(
            f"{lhs_name} and {rhs_name} must share the same shape, "
            f"got {tuple(lhs.shape)} vs {tuple(rhs.shape)}."
        )
