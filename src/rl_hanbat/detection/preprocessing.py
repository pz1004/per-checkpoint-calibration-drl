"""Warm-up-only standardization and common bounded feature mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]


class PreprocessingIntegrityError(RuntimeError):
    """Raised when a stream cannot support locked warm-up preprocessing."""


@dataclass(frozen=True)
class WarmupTransform:
    """Per-feature location and scale estimated from warm-up only."""

    mean: FloatArray
    scale: FloatArray
    ddof: int
    minimum_scale: float

    def transform(self, raw: FloatArray) -> FloatArray:
        """Apply z-scoring and the standard-normal CDF."""
        values = np.asarray(raw, dtype=np.float64)
        if values.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"Expected {self.mean.shape[0]} features, found {values.shape[-1]}"
            )
        if not np.isfinite(values).all():
            raise PreprocessingIntegrityError("Raw detector input is non-finite")
        standardized = (values - self.mean) / self.scale
        bounded = np.asarray(ndtr(standardized), dtype=np.float64)
        if not np.isfinite(bounded).all():
            raise PreprocessingIntegrityError("Bounded detector input is non-finite")
        return bounded


def fit_warmup_transform(
    warmup: FloatArray,
    *,
    expected_windows: int,
    ddof: int,
    minimum_scale: float,
) -> WarmupTransform:
    """Fit a per-stream transform from exactly the declared warm-up matrix."""
    values = np.asarray(warmup, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Warm-up matrix must be two-dimensional, found {values.ndim}")
    if values.shape[0] != expected_windows:
        raise ValueError(
            f"Warm-up matrix has {values.shape[0]} rows; expected {expected_windows}"
        )
    if values.shape[1] != 6:
        raise ValueError(f"Warm-up matrix has {values.shape[1]} features; expected 6")
    if not np.isfinite(values).all():
        raise PreprocessingIntegrityError("Warm-up matrix contains NaN or Inf")
    mean = np.asarray(np.mean(values, axis=0), dtype=np.float64)
    scale = np.asarray(np.std(values, axis=0, ddof=ddof), dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise PreprocessingIntegrityError("Warm-up location or scale is non-finite")
    bad = np.flatnonzero(scale <= minimum_scale)
    if bad.size:
        indices = ", ".join(str(int(index)) for index in bad)
        raise PreprocessingIntegrityError(
            f"Warm-up feature scale is effectively zero at indices {indices}"
        )
    return WarmupTransform(
        mean=mean,
        scale=scale,
        ddof=ddof,
        minimum_scale=minimum_scale,
    )
