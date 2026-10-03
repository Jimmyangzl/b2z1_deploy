"""Normalize scalar or per-joint PD gains from config."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

GainLike = Union[float, int, Sequence[float]]


def as_gain_vector(value: GainLike, n: int, name: str = "gain") -> np.ndarray:
    """Broadcast a scalar or validate a length-``n`` vector to ``float64``."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"{name} must be a scalar or length-{n} list, got {arr.size}")
    return arr.copy()
