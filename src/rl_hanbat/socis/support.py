"""Soft 32-neighbor overlap, mass, and dual-ESS gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from rl_hanbat.socis.config import SOCISConfig


def higher_quantile(values: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        raise ValueError("cannot take a quantile of empty values")
    return float(np.quantile(values, quantile, method="higher"))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(weights**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason_code: str
    supported_mass: float
    current_ess: float
    reference_ess: float
    boundary: float
    current_weights: np.ndarray | None
    reference_weights: np.ndarray | None
    reference_indices: np.ndarray


@dataclass(frozen=True)
class SoftSupportReference:
    """Frozen reference tree and higher 0.99 self-neighbor boundary."""

    x_reference: np.ndarray
    tree: cKDTree
    neighbors: int
    boundary: float
    mass_minimum: float
    current_ess_minimum: float
    reference_ess_minimum: float

    @classmethod
    def fit(cls, config: SOCISConfig, x_reference: np.ndarray) -> SoftSupportReference:
        settings = config.section("support")
        values = np.asarray(x_reference, dtype=np.float64)
        neighbors = int(settings["neighbors"])
        self_neighbors = int(settings["self_query_neighbors"])
        if len(values) < self_neighbors or self_neighbors != neighbors + 1:
            raise ValueError("support reference is too small for self-neighbor query")
        if not np.isfinite(values).all():
            raise ValueError("support reference contains nonfinite values")
        tree = cKDTree(values)
        distances, _ = tree.query(values, k=self_neighbors, workers=1)
        boundary = higher_quantile(
            np.asarray(distances)[:, -1], float(settings["boundary_quantile"])
        )
        if not np.isfinite(boundary) or boundary <= 0.0:
            raise ValueError("support boundary must be finite and positive")
        return cls(
            values,
            tree,
            neighbors,
            boundary,
            float(settings["supported_mass_minimum"]),
            float(settings["current_effective_sample_size_minimum"]),
            float(settings["reference_effective_sample_size_minimum"]),
        )

    def evaluate(self, x_current: np.ndarray) -> SupportResult:
        values = np.asarray(x_current, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("support block must contain finite 2D rows")
        distances, indices = self.tree.query(values, k=self.neighbors, workers=1)
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        rank_distance = distances[:, -1]
        excess = np.maximum(0.0, rank_distance / self.boundary - 1.0)
        coverage = np.exp(-0.5 * excess**2)
        supported_mass = float(np.mean(coverage))
        current_ess = effective_sample_size(coverage)
        reference_mass = np.bincount(
            indices.reshape(-1),
            weights=np.repeat(coverage / self.neighbors, self.neighbors),
            minlength=len(self.x_reference),
        )
        used = np.flatnonzero(reference_mass > 0.0)
        used_mass = reference_mass[used]
        reference_ess = effective_sample_size(used_mass)
        reason = "SUPPORTED"
        if supported_mass < self.mass_minimum:
            reason = "SUPPORTED_MASS_BELOW_MINIMUM"
        elif current_ess < self.current_ess_minimum:
            reason = "CURRENT_ESS_BELOW_MINIMUM"
        elif reference_ess < self.reference_ess_minimum:
            reason = "REFERENCE_ESS_BELOW_MINIMUM"
        supported = reason == "SUPPORTED"
        current_weights = coverage / np.sum(coverage) if supported else None
        reference_weights = used_mass / np.sum(used_mass) if supported else None
        return SupportResult(
            supported=supported,
            reason_code=reason,
            supported_mass=supported_mass,
            current_ess=current_ess,
            reference_ess=reference_ess,
            boundary=self.boundary,
            current_weights=current_weights,
            reference_weights=reference_weights,
            reference_indices=used,
        )


def no_matching_weights(
    current_count: int, reference_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform weights for the prespecified no-matching ablation."""
    if current_count <= 0 or reference_count <= 0:
        raise ValueError("uniform matching requires nonempty groups")
    return (
        np.full(current_count, 1.0 / current_count, dtype=np.float64),
        np.full(reference_count, 1.0 / reference_count, dtype=np.float64),
    )
