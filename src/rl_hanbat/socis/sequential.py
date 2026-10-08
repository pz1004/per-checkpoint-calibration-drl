"""Mode-specific null standardization and abstention-holding GLR scans."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from rl_hanbat.socis.hashing import canonical_sha256
from rl_hanbat.socis.support import higher_quantile


@dataclass(frozen=True)
class NullStandardizer:
    mode: str
    mean: float
    std: float
    source_conditions: tuple[str, ...]
    source_checkpoint_seeds: tuple[int, ...]
    sha256: str

    @classmethod
    def fit(
        cls,
        mode: str,
        scores: np.ndarray,
        *,
        source_conditions: tuple[str, ...],
        source_checkpoint_seeds: tuple[int, ...],
        minimum_std: float = 1.0e-12,
    ) -> NullStandardizer:
        values = np.asarray(scores, dtype=np.float64)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("null standardizer requires finite stable scores")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        if not np.isfinite(std) or std <= minimum_std:
            raise ValueError("null standardizer scale is inert")
        payload = {
            "mode": mode,
            "mean": mean,
            "std": std,
            "source_conditions": list(source_conditions),
            "source_checkpoint_seeds": list(source_checkpoint_seeds),
        }
        return cls(
            mode,
            mean,
            std,
            source_conditions,
            source_checkpoint_seeds,
            canonical_sha256(payload),
        )

    def transform(self, score: float) -> float:
        return (float(score) - self.mean) / self.std


def select_mode_standardizer(
    policy_mode: str, standardizers: dict[str, NullStandardizer]
) -> NullStandardizer:
    """Select only by known frozen/updating mode; no actor value is accepted."""
    if policy_mode not in {"frozen", "updating"}:
        raise ValueError(f"unknown policy mode: {policy_mode}")
    try:
        return standardizers[policy_mode]
    except KeyError as error:
        raise ValueError(f"missing null standardizer for {policy_mode}") from error


@dataclass(frozen=True)
class SequentialTrace:
    statistic: np.ndarray
    alarm: np.ndarray
    first_alarm: int | None


def glr_trace(z_scores: list[float | None], threshold: float) -> SequentialTrace:
    """Apply the unknown-mean GLR; None holds state and advances wall clock."""
    if not np.isfinite(threshold) and threshold != float("inf"):
        raise ValueError("GLR threshold is invalid")
    supported: list[float] = []
    path: list[float] = []
    alarms: list[bool] = []
    first_alarm: int | None = None
    previous = 0.0
    for wall_clock, value in enumerate(z_scores, start=1):
        if value is None:
            statistic = previous
        else:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("GLR score is nonfinite")
            supported.append(numeric)
            cumulative = np.cumsum(np.asarray(supported, dtype=np.float64)[::-1])
            lengths = np.arange(1, len(supported) + 1, dtype=np.float64)
            statistic = float(
                np.max(np.maximum(cumulative, 0.0) ** 2 / (2.0 * lengths))
            )
        crossed = statistic > threshold and first_alarm is None
        if crossed:
            first_alarm = wall_clock
        path.append(statistic)
        alarms.append(crossed)
        previous = statistic
    return SequentialTrace(np.asarray(path), np.asarray(alarms), first_alarm)


def calibrate_glr_threshold(
    stream_scores_by_mode: dict[str, list[list[float | None]]],
    standardizers: dict[str, NullStandardizer],
    *,
    quantile: float = 0.99,
) -> tuple[float, dict[str, float]]:
    """Take the maximum of per-mode higher stream-maximum quantiles."""
    mode_thresholds: dict[str, float] = {}
    for mode in ("frozen", "updating"):
        standardizer = select_mode_standardizer(mode, standardizers)
        maxima: list[float] = []
        for scores in stream_scores_by_mode[mode]:
            z = [
                None if score is None else standardizer.transform(score)
                for score in scores
            ]
            trace = glr_trace(z, float("inf"))
            maxima.append(float(np.max(trace.statistic, initial=0.0)))
        mode_thresholds[mode] = higher_quantile(np.asarray(maxima), quantile)
    return max(mode_thresholds.values()), mode_thresholds


def memoryless_trace(z_scores: list[float | None], threshold: float) -> SequentialTrace:
    """No-accumulation comparator with strict crossing and held abstentions."""
    values: list[float] = []
    alarms: list[bool] = []
    first: int | None = None
    previous = 0.0
    for index, score in enumerate(z_scores, start=1):
        statistic = previous if score is None else max(0.0, float(score))
        crossed = statistic > threshold and first is None
        if crossed:
            first = index
        values.append(statistic)
        alarms.append(crossed)
        previous = statistic
    return SequentialTrace(np.asarray(values), np.asarray(alarms), first)


def adwin_evidence(
    z_scores: list[float | None], *, delta: float = 0.1, minimum_subwindow: int = 5
) -> np.ndarray:
    """Exhaustive deterministic ADWIN-style split evidence on Phi(Z)."""
    observed: list[float] = []
    output: list[float] = []
    normal = NormalDist()
    for score in z_scores:
        if score is None:
            output.append(output[-1] if output else 0.0)
            continue
        observed.append(normal.cdf(float(score)))
        best = 0.0
        count = len(observed)
        for split in range(minimum_subwindow, count - minimum_subwindow + 1):
            left = np.asarray(observed[:split])
            right = np.asarray(observed[split:])
            bound = np.sqrt(
                0.5 * np.log(4.0 / delta) * (1.0 / len(left) + 1.0 / len(right))
            )
            if bound > 0.0:
                best = max(best, abs(float(np.mean(left) - np.mean(right))) / bound)
        output.append(best)
    return np.asarray(output, dtype=np.float64)
