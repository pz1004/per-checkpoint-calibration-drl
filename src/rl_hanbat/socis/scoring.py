"""Fixed PCG64 random features and weighted unbiased off-diagonal MMD."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

from rl_hanbat.socis.config import SOCISConfig
from rl_hanbat.socis.hashing import ndarray_sha256
from rl_hanbat.socis.support import higher_quantile


@dataclass(frozen=True)
class RFFMap:
    frequencies: np.ndarray
    phases: np.ndarray
    bandwidth: float
    sha256: str
    sampled_row_ids: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        config: SOCISConfig,
        residual_reference: np.ndarray,
        row_ids: tuple[str, ...],
    ) -> RFFMap:
        settings = config.section("rff_mmd")
        residuals = np.asarray(residual_reference, dtype=np.float64)
        if residuals.ndim != 2 or residuals.shape[1] < 1:
            raise ValueError("RFF residual reference must be two-dimensional")
        if len(row_ids) != len(residuals) or len(residuals) < 2:
            raise ValueError("RFF row identities do not match residual rows")
        order_keys = np.asarray(
            [
                hashlib.sha256(f"socis-rff-bandwidth-v1|{row_id}".encode()).hexdigest()
                for row_id in row_ids
            ],
            dtype=object,
        )
        order = np.argsort(order_keys, kind="stable")
        limit = min(
            int(settings["bandwidth"]["deterministic_hash_sample_rows"]), len(order)
        )
        selected = order[:limit]
        distances = pdist(residuals[selected], metric="euclidean")
        positive = distances[distances > 0.0]
        if len(positive) == 0:
            raise ValueError("RFF bandwidth is undefined for identical residuals")
        bandwidth = higher_quantile(positive, 0.5)
        feature_count = int(settings["random_features"])
        generator = np.random.Generator(np.random.PCG64(int(settings["rng_seed"])))
        frequencies = generator.normal(size=(feature_count, residuals.shape[1]))
        phases = generator.uniform(0.0, 2.0 * np.pi, size=feature_count)
        digest = ndarray_sha256(
            [frequencies, phases, np.asarray([bandwidth], dtype=np.float64)]
        )
        return cls(
            frequencies,
            phases,
            bandwidth,
            digest,
            tuple(row_ids[int(index)] for index in selected),
        )

    def transform(self, residuals: np.ndarray) -> np.ndarray:
        values = np.asarray(residuals, dtype=np.float64)
        count = len(self.phases)
        return np.sqrt(2.0 / count) * np.cos(
            values @ self.frequencies.T / self.bandwidth + self.phases
        )


def _validate_weights(weights: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("MMD weights have invalid shape or values")
    if np.any(values < 0.0) or not np.isclose(np.sum(values), 1.0, atol=1e-12):
        raise ValueError("MMD weights must be nonnegative and normalized")
    if 1.0 - float(np.sum(values**2)) <= 0.0:
        raise ValueError("MMD group has nonpositive off-diagonal denominator")
    return values


def weighted_unbiased_rff_mmd(
    current_features: np.ndarray,
    reference_features: np.ndarray,
    current_weights: np.ndarray,
    reference_weights: np.ndarray,
) -> float:
    """Compute the untruncated weighted unbiased RFF-MMD formula."""
    current = np.asarray(current_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    if (
        current.ndim != 2
        or reference.ndim != 2
        or current.shape[1] != reference.shape[1]
    ):
        raise ValueError("RFF feature matrices are incompatible")
    a = _validate_weights(current_weights, len(current))
    b = _validate_weights(reference_weights, len(reference))
    mu_a = a @ current
    mu_b = b @ reference
    current_term = (
        float(mu_a @ mu_a) - float(np.sum((a**2) * np.sum(current**2, axis=1)))
    ) / (1.0 - float(np.sum(a**2)))
    reference_term = (
        float(mu_b @ mu_b) - float(np.sum((b**2) * np.sum(reference**2, axis=1)))
    ) / (1.0 - float(np.sum(b**2)))
    return float(current_term + reference_term - 2.0 * float(mu_a @ mu_b))


def reference_rff_moments(
    rff: RFFMap,
    residuals: np.ndarray,
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, float]:
    """Compute fixed-order full-reference feature moments without materializing it."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or len(values) < 2:
        raise ValueError("RFF reference moments require at least two residual rows")
    if batch_size <= 0 or not np.isfinite(values).all():
        raise ValueError("RFF reference moment settings or values differ")
    feature_sum = np.zeros(len(rff.phases), dtype=np.float64)
    squared_norm_sum = 0.0
    for start in range(0, len(values), batch_size):
        features = rff.transform(values[start : start + batch_size])
        feature_sum += np.sum(features, axis=0, dtype=np.float64)
        squared_norm_sum += float(np.sum(features * features, dtype=np.float64))
    return feature_sum, squared_norm_sum


def weighted_unbiased_rff_mmd_uniform_reference_moments(
    current_features: np.ndarray,
    current_weights: np.ndarray,
    reference_feature_sum: np.ndarray,
    reference_squared_norm_sum: float,
    reference_count: int,
) -> float:
    """Exact uniform-reference form using frozen sufficient feature moments."""
    current = np.asarray(current_features, dtype=np.float64)
    reference_sum = np.asarray(reference_feature_sum, dtype=np.float64)
    if (
        current.ndim != 2
        or reference_sum.shape != (current.shape[1],)
        or not np.isfinite(reference_sum).all()
        or not np.isfinite(reference_squared_norm_sum)
        or reference_count < 2
    ):
        raise ValueError("uniform-reference RFF moments differ")
    a = _validate_weights(current_weights, len(current))
    mu_a = a @ current
    mu_b = reference_sum / reference_count
    current_term = (
        float(mu_a @ mu_a) - float(np.sum((a**2) * np.sum(current**2, axis=1)))
    ) / (1.0 - float(np.sum(a**2)))
    inverse = 1.0 / reference_count
    reference_term = (float(mu_b @ mu_b) - inverse**2 * reference_squared_norm_sum) / (
        1.0 - inverse
    )
    return float(current_term + reference_term - 2.0 * float(mu_a @ mu_b))


def direct_weighted_off_diagonal_mmd(
    current_features: np.ndarray,
    reference_features: np.ndarray,
    current_weights: np.ndarray,
    reference_weights: np.ndarray,
) -> float:
    """Direct pair formula used as an independent 1e-12 test oracle."""
    current = np.asarray(current_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    a = _validate_weights(current_weights, len(current))
    b = _validate_weights(reference_weights, len(reference))
    current_kernel = current @ current.T
    reference_kernel = reference @ reference.T
    cross_kernel = current @ reference.T
    current_numerator = float(
        np.sum((a[:, None] * a[None, :]) * current_kernel)
        - np.sum((a**2) * np.diag(current_kernel))
    )
    reference_numerator = float(
        np.sum((b[:, None] * b[None, :]) * reference_kernel)
        - np.sum((b**2) * np.diag(reference_kernel))
    )
    return float(
        current_numerator / (1.0 - float(np.sum(a**2)))
        + reference_numerator / (1.0 - float(np.sum(b**2)))
        - 2.0 * float(a @ cross_kernel @ b)
    )
