"""Shared random-seed helper for Athenan experiments."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs.

    TODO: Extend to torch/JAX once the Athenan backend is selected.
    """

    random.seed(seed)
    np.random.seed(seed)
