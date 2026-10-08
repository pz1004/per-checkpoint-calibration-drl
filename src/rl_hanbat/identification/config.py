"""Configuration loading and locked-protocol validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ProtocolViolation(ValueError):
    """Raised when a scientific value differs from the locked protocol."""


LOCKED_VALUES: dict[tuple[str, ...], Any] = {
    ("protocol", "name"): "hopper_identification_gate",
    ("protocol", "version"): 1,
    ("protocol", "environment_id"): "Hopper-v5",
    ("protocol", "window_size"): 256,
    ("staging", "required_seeds"): list(range(10)),
    ("staging", "integrity_seeds"): [0, 1, 2],
    ("staging", "stage_two_seeds"): list(range(3, 10)),
    ("staging", "auto_expand_stage_two"): True,
    ("sac", "policy"): "MlpPolicy",
    ("sac", "hidden_layers"): [256, 256],
    ("sac", "learning_rate"): 3e-4,
    ("sac", "batch_size"): 256,
    ("sac", "replay_capacity"): 1_000_000,
    ("sac", "learning_starts"): 10_000,
    ("sac", "gamma"): 0.99,
    ("sac", "tau"): 0.005,
    ("sac", "train_frequency"): 1,
    ("sac", "gradient_steps"): 1,
    ("sac", "entropy_coefficient"): "auto",
    ("checkpoint", "preregistered_steps"): 100_000,
    ("checkpoint", "initial_steps"): 300_000,
    ("checkpoint", "extension_steps"): 500_000,
    ("checkpoint", "evaluation_episodes"): 10,
    ("checkpoint", "minimum_median_return"): 500.0,
    ("checkpoint", "probe_states"): 2_048,
    ("branches", "stable_frozen", "windows"): 120,
    ("branches", "stable_frozen", "gravity_multiplier"): 1.0,
    ("branches", "stable_frozen", "updates"): False,
    ("branches", "stable_updating", "windows"): 240,
    ("branches", "stable_updating", "fit_windows"): 120,
    ("branches", "stable_updating", "evaluation_windows"): 120,
    ("branches", "stable_updating", "gravity_multiplier"): 1.0,
    ("branches", "stable_updating", "updates"): True,
    ("branches", "changed_frozen_low", "pre_windows"): 40,
    ("branches", "changed_frozen_low", "post_windows"): 80,
    ("branches", "changed_frozen_low", "updates"): False,
    ("branches", "changed_frozen_low", "gravity_multiplier"): 0.75,
    ("branches", "changed_frozen_high", "pre_windows"): 40,
    ("branches", "changed_frozen_high", "post_windows"): 80,
    ("branches", "changed_frozen_high", "gravity_multiplier"): 1.25,
    ("branches", "changed_frozen_high", "updates"): False,
    ("branch_seed_codes", "checkpoint_train"): 11,
    ("branch_seed_codes", "checkpoint_evaluation"): 13,
    ("branch_seed_codes", "probe_states"): 17,
    ("branch_seed_codes", "stable_frozen"): 101,
    ("branch_seed_codes", "stable_updating"): 103,
    ("branch_seed_codes", "changed_frozen_low"): 107,
    ("branch_seed_codes", "changed_frozen_high"): 107,
    ("nuisance_model", "fit_start_window"): 1,
    ("nuisance_model", "fit_end_window"): 120,
    (
        "nuisance_model",
        "ridge_alphas",
    ): [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    ("nuisance_model", "time_series_splits"): 5,
    ("scoring", "calibration_quantile"): 0.95,
    ("integrity", "active_actor_window_fraction"): 0.90,
    ("integrity", "policy_distance_tolerance"): 1e-10,
    ("integrity", "frozen_hashes_required"): True,
    ("integrity", "finite_values_required"): True,
    ("integrity", "instrumentation_ms_per_step_max"): 5.0,
    ("integrity", "reproducibility_rtol"): 1e-6,
    ("integrity", "reproducibility_atol"): 1e-8,
    ("bootstrap", "replicates"): 2_000,
    ("bootstrap", "block_length_windows"): 8,
    ("bootstrap", "confidence_level"): 0.95,
    ("bootstrap", "seed"): 20_260_731,
    (
        "scientific_gates",
        "update_suppression",
        "median_relative_reduction_min",
    ): 0.30,
    (
        "scientific_gates",
        "update_suppression",
        "nonworse_seed_count_min",
    ): 8,
    (
        "scientific_gates",
        "signal_retention",
        "median_shift_ratio_min",
    ): 0.80,
    (
        "scientific_gates",
        "signal_retention",
        "median_auroc_loss_max",
    ): 0.05,
    (
        "scientific_gates",
        "environment_update_separation",
        "auroc_improvement_min",
    ): 0.05,
    (
        "scientific_gates",
        "environment_update_separation",
        "raw_already_high",
    ): 0.90,
    (
        "scientific_gates",
        "environment_update_separation",
        "high_raw_residual_loss_max",
    ): 0.02,
}


@dataclass(frozen=True)
class IdentificationConfig:
    """Validated experiment configuration and its canonical hash."""

    path: Path
    data: dict[str, Any]
    sha256: str

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level mapping."""
        value = self.data[name]
        if not isinstance(value, dict):
            raise TypeError(f"Configuration section {name!r} is not a mapping")
        return value


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(path)
            raise ProtocolViolation(f"Missing locked configuration value: {joined}")
        current = current[key]
    return current


def canonical_config_bytes(data: dict[str, Any]) -> bytes:
    """Return a deterministic serialization for hashing."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def validate_locked_protocol(data: dict[str, Any]) -> None:
    """Fail if a preregistered scientific value has changed."""
    protocol = data.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("locked") is not True:
        raise ProtocolViolation("protocol.locked must be true")

    mismatches: list[str] = []
    for path, expected in LOCKED_VALUES.items():
        actual = _nested_value(data, path)
        if actual != expected:
            mismatches.append(
                f"{'.'.join(path)}: expected {expected!r}, found {actual!r}"
            )
    if mismatches:
        detail = "\n- ".join(mismatches)
        raise ProtocolViolation(f"Locked protocol mismatch:\n- {detail}")

    branches = data["branches"]
    for branch_name, branch in branches.items():
        expected_updates = branch_name == "stable_updating"
        if branch.get("updates") is not expected_updates:
            raise ProtocolViolation(
                f"branches.{branch_name}.updates must be {expected_updates}"
            )
    for stable_name in ("stable_frozen", "stable_updating"):
        if branches[stable_name].get("gravity_multiplier") != 1.0:
            raise ProtocolViolation(
                f"branches.{stable_name}.gravity_multiplier must be 1.0"
            )


def load_locked_config(path: str | Path) -> IdentificationConfig:
    """Load YAML, validate the locked protocol, and compute its identity."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ProtocolViolation("The configuration root must be a mapping")
    validate_locked_protocol(loaded)
    digest = hashlib.sha256(canonical_config_bytes(loaded)).hexdigest()
    return IdentificationConfig(path=config_path, data=loaded, sha256=digest)
