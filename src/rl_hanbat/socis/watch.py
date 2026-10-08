"""Prespecified blockwise WATCH variant derived from the pinned author tutorial."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from rl_hanbat.socis.config import SOCISConfig


@dataclass(frozen=True)
class WatchBlocks:
    """Block centroids, outcome nonconformity, and dropped-tail evidence."""

    x_centroids: np.ndarray
    outcome_scores: np.ndarray
    block_row_ids: tuple[tuple[str, ...], ...]
    dropped_row_ids: tuple[str, ...]


def aggregate_watch_reference(
    x_standardized: np.ndarray,
    residuals_standardized: np.ndarray,
    row_ids: tuple[str, ...],
    *,
    block_size: int = 256,
    hash_prefix: str = "socis-watch-reference-block-v1",
) -> WatchBlocks:
    """Hash-order reference rows into complete pseudo-blocks and record the tail."""
    x = np.asarray(x_standardized, dtype=np.float64)
    residuals = np.asarray(residuals_standardized, dtype=np.float64)
    if len(x) != len(residuals) or len(row_ids) != len(x):
        raise ValueError("WATCH reference arrays and identities differ")
    if x.ndim != 2 or residuals.ndim != 2 or len(residuals) != len(x):
        raise ValueError("WATCH reference dimensions differ")
    if block_size <= 0 or not np.isfinite(x).all() or not np.isfinite(residuals).all():
        raise ValueError("WATCH reference values or block size differ")
    ranks = np.asarray(
        [
            hashlib.sha256(f"{hash_prefix}|{row_id}".encode()).hexdigest()
            for row_id in row_ids
        ],
        dtype=object,
    )
    order = np.argsort(ranks, kind="stable")
    complete_count = len(order) // block_size
    used = order[: complete_count * block_size]
    dropped = order[complete_count * block_size :]
    if complete_count == 0:
        raise ValueError("WATCH reference has no complete pseudo-block")
    shaped = used.reshape(complete_count, block_size)
    centroids = np.stack([np.mean(x[indices], axis=0) for indices in shaped])
    scores = np.asarray(
        [np.mean(np.sum(residuals[indices] ** 2, axis=1)) for indices in shaped],
        dtype=np.float64,
    )
    blocks = tuple(
        tuple(row_ids[int(index)] for index in indices) for indices in shaped
    )
    tail = tuple(row_ids[int(index)] for index in dropped)
    return WatchBlocks(centroids, scores, blocks, tail)


def aggregate_watch_block(
    x_standardized: np.ndarray, residuals_standardized: np.ndarray
) -> tuple[np.ndarray, float]:
    """Reduce exactly one 256-transition monitoring block to one observation."""
    x = np.asarray(x_standardized, dtype=np.float64)
    residuals = np.asarray(residuals_standardized, dtype=np.float64)
    if (x.ndim != 2 or residuals.ndim != 2 or len(residuals) != len(x)
            or len(x) != 256):
        raise ValueError("WATCH monitoring block must contain exactly 256 rows")
    if not np.isfinite(x).all() or not np.isfinite(residuals).all():
        raise ValueError("WATCH monitoring block contains nonfinite values")
    return np.mean(x, axis=0), float(np.mean(np.sum(residuals**2, axis=1)))


def randomized_rank_p_value(
    reference_scores: np.ndarray,
    current_score: float,
    *,
    uniform: float,
    weights: np.ndarray | None = None,
) -> float:
    """Tutorial rank: strict-greater mass plus U times tied mass including current."""
    reference = np.asarray(reference_scores, dtype=np.float64)
    if (
        reference.ndim != 1
        or not np.isfinite(reference).all()
        or not np.isfinite(current_score)
    ):
        raise ValueError("WATCH scores must be finite and one-dimensional")
    scores = np.concatenate([reference, [float(current_score)]])
    if not 0.0 <= uniform < 1.0:
        raise ValueError("tie uniform must lie in [0, 1)")
    if weights is None:
        normalized = np.full(len(scores), 1.0 / len(scores), dtype=np.float64)
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        if (
            normalized.shape != scores.shape
            or not np.isfinite(normalized).all()
            or np.any(normalized < 0.0)
        ):
            raise ValueError("WATCH weights differ from score rows")
        total = float(np.sum(normalized))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("WATCH weights have invalid mass")
        normalized = normalized / total
    greater = float(np.sum(normalized[scores > current_score]))
    tied = float(np.sum(normalized[scores == current_score]))
    return greater + uniform * tied


def tutorial_conformal_p_values(scores: np.ndarray, *, seed: int) -> np.ndarray:
    """Reproduce the author tutorial's sequential strict-rank primitive."""
    values = np.asarray(scores, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    p_values: list[float] = []
    for index, score in enumerate(values):
        prefix = values[:index]
        p_values.append(
            randomized_rank_p_value(
                prefix, float(score), uniform=float(generator.uniform())
            )
        )
    return np.asarray(p_values, dtype=np.float64)


@dataclass
class CompositeJumper:
    """Exact five-J mixture used at author revision 3d17fca..."""

    jump_rates: np.ndarray
    minus: np.ndarray
    zero: np.ndarray
    plus: np.ndarray
    wealth: float

    @classmethod
    def create(
        cls, jump_rates: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0)
    ) -> CompositeJumper:
        rates = np.asarray(jump_rates, dtype=np.float64)
        if (
            rates.ndim != 1
            or len(rates) == 0
            or not np.isfinite(rates).all()
            or np.any(rates < 0.0)
            or np.any(rates > 1.0)
        ):
            raise ValueError("composite-jumper rates must lie in [0, 1]")
        initial = np.full(len(rates), 1.0 / 3.0, dtype=np.float64)
        return cls(rates, initial.copy(), initial.copy(), initial.copy(), 1.0)

    def update(self, p_value: float) -> float:
        if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError("composite-jumper p-value is invalid")
        for index, jump in enumerate(self.jump_rates):
            self.minus[index] = (1.0 - jump) * self.minus[index] + (
                jump / 3.0
            ) * self.wealth
            self.zero[index] = (1.0 - jump) * self.zero[index] + (
                jump / 3.0
            ) * self.wealth
            self.plus[index] = (1.0 - jump) * self.plus[index] + (
                jump / 3.0
            ) * self.wealth
            self.minus[index] *= 1.0 - (p_value - 0.5)
            self.plus[index] *= 1.0 + (p_value - 0.5)
        self.wealth = float(np.mean(self.minus + self.zero + self.plus))
        return self.wealth


def tutorial_composite_jumper(p_values: np.ndarray) -> np.ndarray:
    """Return the pinned tutorial wealth path for a p-value sequence."""
    jumper = CompositeJumper.create()
    return np.asarray([jumper.update(float(value)) for value in p_values])


@dataclass(frozen=True)
class WatchStep:
    x_p_value: float
    x_wealth: float
    adaptation_active: bool
    outcome_p_value: float
    outcome_wealth: float


class BlockwiseWatchAdapter:
    """Causal blockwise WATCH variant with an internal strict X-CTM trigger."""

    def __init__(self, config: SOCISConfig, reference: WatchBlocks) -> None:
        settings = config.section("comparators")["watch"]
        self.reference = reference
        if (
            reference.x_centroids.ndim != 2
            or len(reference.x_centroids) == 0
            or reference.outcome_scores.shape != (len(reference.x_centroids),)
            or not np.isfinite(reference.x_centroids).all()
            or not np.isfinite(reference.outcome_scores).all()
        ):
            raise ValueError("WATCH adapter reference differs")
        self.settings = settings
        self.reference_centroid = np.mean(reference.x_centroids, axis=0)
        self.reference_x_scores = np.sum(
            (reference.x_centroids - self.reference_centroid) ** 2, axis=1
        )
        self.x_jumper = CompositeJumper.create(
            tuple(float(value) for value in settings["composite_jumper"]["jump_rates"])
        )
        self.outcome_jumper = CompositeJumper.create(
            tuple(float(value) for value in settings["composite_jumper"]["jump_rates"])
        )
        seed = int(settings["conformal_rank"]["tie_break_seed"])
        children = np.random.SeedSequence(seed).spawn(2)
        self.x_rng = np.random.Generator(np.random.PCG64(children[0]))
        self.outcome_rng = np.random.Generator(np.random.PCG64(children[1]))
        self.previous_centroids: list[np.ndarray] = []
        self.adaptation_active = False

    def _density_ratio_weights(self, current: np.ndarray) -> np.ndarray | None:
        if not self.adaptation_active or not self.previous_centroids:
            return None
        ratio = self.settings["density_ratio"]
        source = self.reference.x_centroids
        target = np.stack(self.previous_centroids)
        training = np.concatenate([source, target])
        labels = np.concatenate(
            [np.zeros(len(source), dtype=np.int8), np.ones(len(target), dtype=np.int8)]
        )
        classifier = LogisticRegression(
            C=float(ratio["c"]),
            solver=str(ratio["solver"]),
            max_iter=int(ratio["maximum_iterations"]),
            tol=float(ratio["tolerance"]),
            class_weight=None,
            random_state=int(ratio["random_state"]),
        )
        classifier.fit(training, labels)
        evaluated = np.concatenate([source, current[None, :]])
        probability = classifier.predict_proba(evaluated)[:, 1]
        clip = float(ratio["probability_clip"])
        probability = np.clip(probability, clip, 1.0 - clip)
        weights = probability / (1.0 - probability)
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("WATCH density-ratio weights are invalid")
        return weights

    def update(self, x_centroid: np.ndarray, outcome_score: float) -> WatchStep:
        current = np.asarray(x_centroid, dtype=np.float64)
        if (
            current.shape != self.reference.x_centroids.shape[1:]
            or not np.isfinite(current).all()
        ):
            raise ValueError("WATCH current centroid dimension differs")
        if not np.isfinite(outcome_score):
            raise ValueError("WATCH outcome score is nonfinite")
        x_score = float(np.sum((current - self.reference_centroid) ** 2))
        x_p = randomized_rank_p_value(
            self.reference_x_scores,
            x_score,
            uniform=float(self.x_rng.uniform()),
        )
        x_wealth = self.x_jumper.update(x_p)
        trigger = float(self.settings["x_ctm"]["strict_trigger_wealth"])
        if x_wealth > trigger:
            self.adaptation_active = True
        weights = self._density_ratio_weights(current)
        outcome_p = randomized_rank_p_value(
            self.reference.outcome_scores,
            outcome_score,
            uniform=float(self.outcome_rng.uniform()),
            weights=weights,
        )
        outcome_wealth = self.outcome_jumper.update(outcome_p)
        step = WatchStep(
            x_p,
            x_wealth,
            self.adaptation_active,
            outcome_p,
            outcome_wealth,
        )
        self.previous_centroids.append(current.copy())
        return step


TUTORIAL_GOLDEN_SCORES = np.asarray([0.2, 0.1, 0.1, 0.4, 0.05], dtype=np.float64)
TUTORIAL_GOLDEN_SEED = 7
# Frozen from the formulas in WATCH_TutorialExample.ipynb and src/martingales.py
# at revision 3d17fca58ac60410d020b54dee19ee3fc4ec29c2.
TUTORIAL_GOLDEN_P_VALUES = np.asarray(
    [
        0.625095466604667,
        0.9486069004847877,
        0.8504571268301289,
        0.056301797497647965,
        0.8600332569822451,
    ],
    dtype=np.float64,
)
TUTORIAL_GOLDEN_WEALTH = np.asarray(
    [
        1.0,
        1.0290986628993566,
        1.1384715001413863,
        0.9471508527953294,
        0.9748064854199316,
    ],
    dtype=np.float64,
)
