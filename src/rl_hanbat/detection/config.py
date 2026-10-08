"""Configuration loading and validation for the detector benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_hanbat.identification.features import TRAJECTORY_FEATURES


class DetectorProtocolViolation(ValueError):
    """Raised when a scientific benchmark setting differs from the protocol."""


LOCKED_VALUES: dict[tuple[str, ...], Any] = {
    ("protocol", "name"): "hopper_raw_feature_detector_benchmark",
    ("protocol", "version"): 1,
    ("protocol", "environment_id"): "Hopper-v5",
    ("protocol", "run_root"): "artifacts/detector_benchmark",
    ("protocol", "window_size"): 256,
    ("protocol", "raw_features"): list(TRAJECTORY_FEATURES),
    ("source", "expected_seeds"): list(range(10)),
    (
        "source",
        "qualified_record_pattern",
    ): "seeds/seed_{seed}/checkpoint/qualified/checkpoint.json",
    ("source", "manifest_name"): "manifest.json",
    ("staging", "integrity_seeds"): [0, 1, 2],
    ("staging", "expansion_seeds"): list(range(3, 10)),
    ("staging", "auto_expand_calibration"): True,
    ("streams", "splits"): ["calibration", "holdout"],
    ("streams", "replicates_per_seed_condition"): 4,
    ("streams", "warmup_windows"): 40,
    ("streams", "monitoring_windows"): 80,
    ("streams", "expected_streams"): 320,
    ("streams", "expected_windows"): 38_400,
    ("streams", "expected_transitions"): 9_830_400,
    ("stream_seed_codes", "calibration"): 401,
    ("stream_seed_codes", "holdout"): 409,
    ("stream_seed_codes", "frozen"): 419,
    ("stream_seed_codes", "updating"): 421,
    ("stream_seed_codes", "reproducibility"): 431,
    ("execution", "device"): "auto",
    ("preprocessing", "standard_deviation_ddof"): 1,
    ("preprocessing", "minimum_scale"): 1e-12,
    ("preprocessing", "bounded_map"): "standard_normal_cdf",
    ("detectors", "feature_aggregation"): "maximum",
    ("detectors", "memoryless", "settings"): ["default"],
    (
        "detectors",
        "memoryless",
        "evidence",
    ): "two_sided_bounded_deviation",
    ("detectors", "cusum", "drift_grid"): [0.0, 0.025, 0.05, 0.10, 0.20],
    ("detectors", "cusum", "center"): 0.5,
    ("detectors", "cusum", "sidedness"): "two_sided",
    ("detectors", "adwin", "delta_grid"): [
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
    ],
    ("detectors", "adwin", "minimum_subwindow_length"): 5,
    ("detectors", "adwin", "cut_bound"): "bounded_hoeffding",
    ("detectors", "adwin", "exhaustive_splits"): True,
    ("detectors", "bocpd", "expected_run_length_grid"): [40, 80, 160, 320],
    ("detectors", "bocpd", "changepoint_run_lengths"): [0, 1, 2, 3, 4],
    ("detectors", "bocpd", "prior", "mu"): 0.0,
    ("detectors", "bocpd", "prior", "kappa"): 1.0,
    ("detectors", "bocpd", "prior", "alpha"): 2.0,
    ("detectors", "bocpd", "prior", "beta"): 0.25,
    ("thresholds", "primary_quantile"): 0.95,
    ("thresholds", "quantile_method"): "higher",
    ("thresholds", "stable_condition_rule"): "maximum",
    ("thresholds", "stopping_comparison"): "strict_greater",
    ("selection", "miss_restricted_delay_windows"): 81,
    ("selection", "transitions_per_delay_window"): 256,
    (
        "selection",
        "setting_objective",
    ): "worst_direction_median_restricted_delay",
    (
        "selection",
        "setting_tie_breaks",
    ): [
        "average_direction_median_restricted_delay",
        "declared_grid_order",
    ],
    ("selection", "eligibility_false_alarm_max"): 0.05,
    ("selection", "sequential_order"): ["cusum", "adwin", "bocpd"],
    ("bootstrap", "replicates"): 10_000,
    ("bootstrap", "confidence_level"): 0.95,
    ("bootstrap", "seed"): 20_260_731,
    ("bootstrap", "method"): "paired_seed_hierarchical_max_statistic",
    ("bootstrap", "seed_count"): 10,
    ("bootstrap", "within_seed_replicates"): 4,
    ("integrity", "active_actor_window_fraction"): 0.90,
    ("integrity", "policy_distance_tolerance"): 1e-10,
    ("integrity", "finite_values_required"): True,
    ("integrity", "instrumentation_ms_per_step_max"): 5.0,
    ("integrity", "reproducibility_rtol"): 1e-6,
    ("integrity", "reproducibility_atol"): 1e-8,
    ("integrity", "require_freeze_before_holdout"): True,
    (
        "integrity",
        "verify_checkpoint_files",
    ): ["model", "replay_buffer", "rng_state", "probe_states"],
    (
        "integrity",
        "verify_checkpoint_components",
    ): ["actor", "critics", "target_critics"],
}


@dataclass(frozen=True)
class DetectorConfig:
    """Validated detector configuration and canonical identity."""

    path: Path
    data: dict[str, Any]
    sha256: str

    def section(self, name: str) -> dict[str, Any]:
        """Return one top-level configuration mapping."""
        value = self.data[name]
        if not isinstance(value, dict):
            raise TypeError(f"Configuration section {name!r} is not a mapping")
        return value


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise DetectorProtocolViolation(
                f"Missing locked configuration value: {'.'.join(path)}"
            )
        current = current[key]
    return current


def canonical_json_bytes(data: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for hashing."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def validate_detector_protocol(data: dict[str, Any]) -> None:
    """Fail closed when a locked scientific or integrity value changes."""
    protocol = data.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("locked") is not True:
        raise DetectorProtocolViolation("protocol.locked must be true")

    mismatches: list[str] = []
    for path, expected in LOCKED_VALUES.items():
        actual = _nested_value(data, path)
        if actual != expected:
            mismatches.append(
                f"{'.'.join(path)}: expected {expected!r}, found {actual!r}"
            )
    if mismatches:
        raise DetectorProtocolViolation(
            "Locked detector protocol mismatch:\n- " + "\n- ".join(mismatches)
        )

    conditions = _nested_value(data, ("streams", "conditions"))
    if not isinstance(conditions, dict):
        raise DetectorProtocolViolation("streams.conditions must be a mapping")
    expected_conditions = {
        "stable_frozen": (1.0, False, False),
        "stable_updating": (1.0, True, False),
        "changed_frozen_low": (0.75, False, True),
        "changed_frozen_high": (1.25, False, True),
    }
    for name, expected in expected_conditions.items():
        condition = conditions.get(name)
        if not isinstance(condition, dict):
            raise DetectorProtocolViolation(f"Missing condition {name}")
        actual = (
            condition.get("gravity_multiplier"),
            condition.get("updates"),
            condition.get("change_after_warmup"),
        )
        if actual != expected:
            raise DetectorProtocolViolation(
                f"Condition {name} must be gravity/updates/change={expected}"
            )

    quantiles = _nested_value(data, ("thresholds", "operating_curve_quantiles"))
    expected_quantiles = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.975,
        0.99,
        1.00,
    ]
    if quantiles != expected_quantiles:
        raise DetectorProtocolViolation(
            "thresholds.operating_curve_quantiles differs from the protocol"
        )


def load_detector_config(path: str | Path) -> DetectorConfig:
    """Load YAML, validate every lock, and compute its canonical hash."""
    config_path = Path(path).expanduser().resolve()
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise DetectorProtocolViolation("The configuration root must be a mapping")
    validate_detector_protocol(loaded)
    digest = hashlib.sha256(canonical_json_bytes(loaded)).hexdigest()
    return DetectorConfig(path=config_path, data=loaded, sha256=digest)
