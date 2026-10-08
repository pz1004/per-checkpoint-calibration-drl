"""Validation of the implemented-but-disabled SOCIS-RL protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_hanbat.socis.hashing import file_sha256, zero_field_identity


class SOCISProtocolViolation(ValueError):
    """Raised when a scientific, seed, count, or execution lock differs."""


CONDITIONS = (
    "stable_frozen",
    "stable_updating",
    "changed_frozen_low",
    "changed_frozen_high",
    "changed_updating_low",
    "changed_updating_high",
)
STABLE_CONDITIONS = ("stable_frozen", "stable_updating")
CHANGED_CONDITIONS = CONDITIONS[2:]
DEVELOPMENT_SEEDS = tuple(range(30, 40))
CONFIRMATION_SEEDS = tuple(range(40, 60))
FORBIDDEN_LEGACY_SEEDS = tuple(range(10, 30))
WALKER_DEVELOPMENT_SEEDS = tuple(range(60, 70))
WALKER_CONFIRMATION_SEEDS = tuple(range(70, 90))
HALFCHEETAH_DEVELOPMENT_SEEDS = tuple(range(90, 100))
HALFCHEETAH_CONFIRMATION_SEEDS = tuple(range(100, 120))
CONFIRMATION_METHODS = (
    "socis_rl",
    "watch",
    "adwin",
    "memoryless",
    "no_matching",
    "no_accumulation",
)
WATCH_REVISION = "3d17fca58ac60410d020b54dee19ee3fc4ec29c2"


def _mapping(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise SOCISProtocolViolation(f"{name} must be a mapping")
    return value


def _integers(section: dict[str, Any], name: str) -> list[int]:
    value = section.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise SOCISProtocolViolation(f"{name} must be a list of integers")
    return value


def _expect(actual: Any, expected: Any, name: str) -> None:
    boolean_mismatch = isinstance(expected, bool) and not isinstance(actual, bool)
    integer_boolean_mismatch = (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and isinstance(actual, bool)
    )
    if actual != expected or boolean_mismatch or integer_boolean_mismatch:
        raise SOCISProtocolViolation(
            f"{name} must be {expected!r}; observed {actual!r}"
        )


def _expect_values(
    section: dict[str, Any], expected: dict[str, Any], prefix: str
) -> None:
    for key, value in expected.items():
        _expect(section.get(key), value, f"{prefix}.{key}")


def _validate_protocol_header(data: dict[str, Any]) -> None:
    """Reject changes to the amended primary protocol header."""
    _expect_values(
        _mapping(data, "protocol"),
        {
            "amendment": 2,
            "operational_revision": 3,
            "amended_date": "2026-08-05",
            "amendment_1_sha256": (
                "0c337ad2c1ae64d83a424a937cde51ae034a8ce40ee1ca09b08dfd9dd90535e9"
            ),
            "amendment_sha256": (
                "d27e9163ade0050a611ba44b64a48e64cb6ea0933d00ffc05d359c111401023d"
            ),
            "phase_c_operational_lock_sha256": (
                "2ac06e9ff4f4c9e8c99273cdb11a5576d57638aa7d9e9120a61c89b0bcad6322"
            ),
            "phase_d_operational_lock_sha256": (
                "18f0a69132520d42030cf89c388ab6e9e93e1510ade77d56e6127420d55727ea"
            ),
            "environment_id": "Hopper-v5",
            "learner": "SAC",
            "scientific_scope": "detector_validity_only",
            "claim_scope": "fixed_horizon_empirical_calibration",
            "run_root": "artifacts/socis_rl",
        },
        "protocol",
    )


def _validate_confirmation_analysis(confirmation: dict[str, Any]) -> None:
    analysis = _mapping(confirmation, "analysis")
    _expect_values(
        analysis,
        {
            "methods": list(CONFIRMATION_METHODS),
            "expected_streams_per_condition": 100,
            "inferential_cluster": "checkpoint_seed",
            "within_seed_replicates": 5,
            "fixed_horizon_windows": 80,
            "miss_restricted_delay": 81,
            "familywise_alpha": 0.05,
            "success_rule": "intersection_all_integrity_and_scientific_gates",
            "positive_outcome": "POSITIVE_CONFIRMATION",
            "scientific_failure_outcome": "NEGATIVE_CONFIRMATION_STOP",
            "integrity_failure_outcome": "CONFIRMATION_INVALID",
            "intervals_are_population_guarantees": False,
        },
        "confirmation.analysis",
    )


def _validate_phase_c_operational(data: dict[str, Any]) -> None:
    """Validate Phase C operations and the remaining confirmation detail locks."""
    _expect_values(
        _mapping(data, "checkpoint_contract"),
        {
            "checkpoint_manifest_required": True,
            "checkpoint_manifest_filename": "socis_checkpoint_manifest_v2.json",
            "checkpoint_manifest_schema_version": 2,
            "qualified_stable_replay_required": True,
            "policy_hidden_layers": [256, 256],
            "required_checkpoint_files": [
                "model",
                "replay_buffer",
                "rng_state",
                "probe_states",
            ],
            "required_component_hashes": [
                "actor",
                "critics",
                "target_critics",
            ],
            "training_transitions_excluded_from_monitoring_counts": True,
            "checkpoint_creation_requires_separate_review": True,
        },
        "checkpoint_contract",
    )
    training = _mapping(data, "checkpoint_training")
    _expect_values(
        training,
        {
            "status": "implemented_not_authorized",
            "environment_id": "Hopper-v5",
            "gravity_multiplier": 1.0,
            "policy": "MlpPolicy",
            "hidden_layers": [256, 256],
            "activation": "relu",
            "optimizer": "adam",
            "optimizer_epsilon": 1.0e-5,
            "learning_rate": 3.0e-4,
            "learning_rate_schedule": "constant",
            "batch_size": 256,
            "replay_capacity": 1_000_000,
            "learning_starts": 10_000,
            "gamma": 0.99,
            "tau": 0.005,
            "train_frequency_steps": 1,
            "gradient_steps": 1,
            "target_update_interval": 1,
            "entropy_coefficient": "auto",
            "target_entropy": "auto",
            "action_noise": None,
            "use_sde": False,
            "optimize_memory_usage": False,
            "observation_normalization": False,
            "reward_normalization": False,
            "device_when_authorized": "cuda:0",
        },
        "checkpoint_training",
    )
    _expect(
        _mapping(training, "seed_codes"),
        {"training": 11, "evaluation": 13, "probe_states": 17},
        "checkpoint_training.seed_codes",
    )
    _expect_values(
        _mapping(training, "qualification"),
        {
            "retained_stage_steps": 100_000,
            "initial_evaluation_steps": 300_000,
            "maximum_extension_steps": 500_000,
            "evaluation_episodes": 10,
            "deterministic_actions": True,
            "minimum_median_return": 500.0,
            "finite_trajectory_required": True,
            "probe_states": 2_048,
            "replacement_seed_allowed": False,
            "further_extension_allowed": False,
        },
        "checkpoint_training.qualification",
    )
    _expect_values(
        _mapping(training, "stable_replay"),
        {
            "schema_version": 1,
            "ordering": "ascending_physical_buffer_index",
            "condition": "stable_frozen",
            "observation_dimension": 11,
            "action_dimension": 3,
            "continuous_dtype": "float64",
            "parquet_compression": "zstd",
            "every_occupied_row_required": True,
            "changed_or_monitoring_rows_allowed": False,
        },
        "checkpoint_training.stable_replay",
    )
    _expect_values(
        _mapping(training, "resume"),
        {
            "completed_seed_behavior": "verify_without_regeneration",
            "interrupted_seed_behavior": ("preserve_then_restart_from_initial_seed"),
            "continue_unverified_mid_training_state": False,
            "completed_seed_overwrite_allowed": False,
            "manifest_publication": "atomic_no_clobber",
        },
        "checkpoint_training.resume",
    )
    authorization = _mapping(data, "phase_c_authorization")
    _expect_values(
        authorization,
        {
            "schema_version": 1,
            "purpose": "socis_phase_c_checkpoint_training",
            "decision": "AUTHORIZE_PHASE_C",
            "canonical_command": ("python -m rl_hanbat.socis train-checkpoints"),
            "checkpoint_seeds": list(DEVELOPMENT_SEEDS),
            "device": "cuda:0",
            "output_root_parent": "artifacts/socis_rl",
            "output_root_name_pattern": (
                "^phase_c_checkpoints_[a-z0-9][a-z0-9_-]{0,47}$"
            ),
            "expiry_required": True,
            "human_approver_required": True,
            "rationale_required": True,
            "zero_field": "self_sha256",
            "authorization_scope": "checkpoint_creation_and_verification_only",
            "development_collection_authorized": False,
            "confirmation_authorized": False,
        },
        "phase_c_authorization",
    )
    phase_d = _mapping(data, "phase_d_authorization")
    checkpoint_source_root = phase_d.get("checkpoint_source_root")
    # Preserve the exact historical path lock without publishing the private path.
    if (
        not isinstance(checkpoint_source_root, str)
        or hashlib.sha256(checkpoint_source_root.encode("utf-8")).hexdigest()
        != "4d78534e8d1dd2a2cbe8654b12a073a6fa955177f201f204b957e472ce1a047f"
    ):
        raise SOCISProtocolViolation(
            "phase_d_authorization.checkpoint_source_root differs from locked value"
        )
    _expect_values(
        phase_d,
        {
            "schema_version": 1,
            "purpose": "socis_phase_d_development",
            "decision": "AUTHORIZE_PHASE_D",
            "canonical_command": "python -m rl_hanbat.socis develop",
            "checkpoint_seeds": list(DEVELOPMENT_SEEDS),
            "device": "cuda:0",
            "checkpoint_manifest_sha256": (
                "61dff539314bf9a61583baea5aa924d76150bc62c538b4177fbde4fc92574da3"
            ),
            "checkpoint_authorization_sha256": (
                "f7965de653134e4fe2ef2122a7f5d24bc1bbeb8f0511f77997c04809556edc47"
            ),
            "gate_1_source_tree_sha256": (
                "7cfae66779844fc1e9773c66c3cf7e89f8503fbc0f5aeaf7548d969285af2b58"
            ),
            "gate_1_config_sha256": (
                "0188a05614984c7e30d1e83b11b4e3fd108b9ea3225a1d863abd942bc665008e"
            ),
            "gate_1_config_raw_sha256": (
                "c148278d1d21c820b4ffc4cc3cd7f39b38b514cd4ae7db2cdd7588a620462f44"
            ),
            "output_root_parent": "artifacts/socis_rl",
            "output_root_name": "phase_d_development_primary_v1",
            "expiry_required": True,
            "delegated_approver_required": True,
            "rationale_required": True,
            "terminal_scientific_failure_acceptance_required": True,
            "zero_field": "self_sha256",
            "authorization_scope": "stable_first_development_only",
            "development_collection_authorized": True,
            "confirmation_authorized": False,
        },
        "phase_d_authorization",
    )
    interface = _mapping(data, "interface")
    _expect_values(
        interface,
        {
            "commands": [
                "train-checkpoints",
                "verify-checkpoints",
                "develop",
                "verify-freeze",
                "confirm",
            ]
        },
        "interface",
    )
    _expect_values(
        _mapping(interface, "train_checkpoints"),
        {
            "config_argument_required": True,
            "authorization_argument_required": True,
            "output_root_argument_required": True,
            "exact_seed_panel_only": True,
            "stop_after_checkpoint_qualification": True,
        },
        "interface.train_checkpoints",
    )
    _expect_values(
        _mapping(interface, "develop"),
        {
            "config_argument_required": True,
            "authorization_argument_required": True,
            "source_run_directory_argument_required": True,
            "run_directory_argument_required": True,
            "stop_after_development": True,
        },
        "interface.develop",
    )
    _expect_values(
        _mapping(interface, "verify_checkpoints"),
        {
            "read_only": True,
            "config_argument_required": True,
            "authorization_argument_required": True,
            "output_root_argument_required": True,
        },
        "interface.verify_checkpoints",
    )
    analysis = _mapping(_mapping(data, "confirmation"), "analysis")
    _expect_values(
        _mapping(analysis, "false_alarm"),
        {
            "stable_conditions": list(STABLE_CONDITIONS),
            "all_method_point_maximum_per_cell": 5,
            "primary_method": "socis_rl",
            "primary_exact_bound": "clopper_pearson_one_sided_upper",
            "primary_bonferroni_cells": 2,
            "primary_upper_bound_maximum": 0.10,
            "implied_primary_maximum_count": 4,
        },
        "confirmation.analysis.false_alarm",
    )
    _expect_values(
        _mapping(analysis, "detection"),
        {
            "conditions": list(CHANGED_CONDITIONS),
            "primary_method": "socis_rl",
            "exact_bound": "clopper_pearson_one_sided_lower",
            "bonferroni_cells": 4,
            "lower_bound_minimum": 0.50,
            "implied_minimum_count": 62,
        },
        "confirmation.analysis.detection",
    )
    _expect_values(
        _mapping(analysis, "support"),
        {
            "primary_method": "socis_rl",
            "stable_eligible_block_fraction_minimum_per_mode": 0.90,
            "changed_eligible_block_fraction_minimum_per_regime": 0.80,
            "accepted_block_supported_mass_minimum": 0.5,
            "accepted_block_current_ess_minimum": 128,
            "accepted_block_reference_ess_minimum": 128,
        },
        "confirmation.analysis.support",
    )
    _expect_values(
        _mapping(analysis, "delay"),
        {
            "condition": "changed_updating_low",
            "effect": "comparator_minus_socis_mean_restricted_delay",
            "comparator_order": [
                "watch",
                "adwin",
                "memoryless",
                "no_matching",
                "no_accumulation",
            ],
            "aggregate_replicates_within_seed": "mean",
            "simultaneous_method": "studentized_rademacher_wild_cluster",
            "simultaneous_confidence": 0.95,
            "bootstrap_draws": 50_000,
            "bootstrap_seed": 20260807,
            "noninferiority_lower_bound_methods": ["watch", "adwin"],
            "strict_superiority_lower_bound_methods": [
                "memoryless",
                "no_matching",
                "no_accumulation",
            ],
            "memoryless_relative_point_improvement_minimum": 0.30,
        },
        "confirmation.analysis.delay",
    )
    _expect_values(
        _mapping(analysis, "missingness"),
        {
            "final_missing_or_invalid_shard": "confirmation_invalid",
            "imputation_allowed": False,
            "complete_case_subset_allowed": False,
            "replacement_seed_allowed": False,
            "interrupted_shard_behavior": "resume_identically",
        },
        "confirmation.analysis.missingness",
    )


def _validate_external_and_target(data: dict[str, Any]) -> None:
    """Reject changes to external studies, seed walls, and the target."""
    external = _mapping(data, "external_validity")
    _expect_values(
        external,
        {
            "status": "preregistered_not_implemented",
            "execution_contingent_on_positive_primary_confirmation": True,
            "required_for_general_continuous_control_claim": True,
            "selection_locked_before_primary_outcomes": True,
            "broad_claim_success_rule": "primary_and_both_external_studies_pass",
            "mixed_result_interpretation": "heterogeneity_not_general_robustness",
        },
        "external_validity",
    )
    common = _mapping(external, "common")
    _expect_values(
        common,
        {
            "learner": "SAC",
            "development_replicates": [0, 1, 2, 3],
            "confirmation_replicates": [0, 1, 2, 3, 4],
            "conditions": list(CONDITIONS),
            "development_streams": 240,
            "confirmation_streams": 600,
            "warmup_windows": 40,
            "monitoring_windows": 80,
            "window_size": 256,
            "unchanged_method_hyperparameters_required": True,
            "stable_only_calibration_required": True,
            "use_primary_development_gates": True,
            "use_amendment_1_confirmation_rule": True,
        },
        "external_validity.common",
    )
    studies = _mapping(external, "studies")
    if set(studies) != {"ev_walker_gravity", "ev_halfcheetah_mass"}:
        raise SOCISProtocolViolation("external validity study set differs")
    expected_studies = {
        "ev_walker_gravity": {
            "environment_id": "Walker2d-v5",
            "perturbation": "gravity_multiplier",
            "low_multiplier": 0.75,
            "high_multiplier": 1.25,
            "application_time": "after_warmup",
            "observation_dimension": 17,
            "action_dimension": 6,
            "residual_dimension": 20,
            "development_checkpoint_seeds": list(WALKER_DEVELOPMENT_SEEDS),
            "confirmation_checkpoint_seeds": list(WALKER_CONFIRMATION_SEEDS),
        },
        "ev_halfcheetah_mass": {
            "environment_id": "HalfCheetah-v5",
            "perturbation": "all_non_world_body_mass_multiplier",
            "low_multiplier": 0.75,
            "high_multiplier": 1.25,
            "application_time": "after_warmup",
            "observation_dimension": 17,
            "action_dimension": 6,
            "residual_dimension": 20,
            "development_checkpoint_seeds": list(HALFCHEETAH_DEVELOPMENT_SEEDS),
            "confirmation_checkpoint_seeds": list(HALFCHEETAH_CONFIRMATION_SEEDS),
        },
    }
    for name, expected in expected_studies.items():
        study = studies.get(name)
        if not isinstance(study, dict):
            raise SOCISProtocolViolation(f"external validity {name} differs")
        _expect(study, expected, f"external_validity.studies.{name}")
    all_seed_sets = (
        FORBIDDEN_LEGACY_SEEDS,
        DEVELOPMENT_SEEDS,
        CONFIRMATION_SEEDS,
        WALKER_DEVELOPMENT_SEEDS,
        WALKER_CONFIRMATION_SEEDS,
        HALFCHEETAH_DEVELOPMENT_SEEDS,
        HALFCHEETAH_CONFIRMATION_SEEDS,
    )
    union: set[int] = set()
    for seed_set in all_seed_sets:
        if union & set(seed_set):
            raise SOCISProtocolViolation("primary and external seed walls overlap")
        union.update(seed_set)
    _expect_values(
        _mapping(data, "target"),
        {
            "covariates": ["observation", "action"],
            "outcomes": ["state_delta", "reward", "terminated", "truncated"],
            "estimand": "conditional_outcome_law_on_shared_support",
            "occupancy_only_change_in_scope": False,
            "actor_change_residualization_allowed": False,
            "policy_mode_may_select_null_standardizer": True,
        },
        "target",
    )


def _validate_publication_and_science(data: dict[str, Any]) -> None:
    """Reject changes to publication evidence and remaining science locks."""
    publication = _mapping(data, "publication_evidence")
    _expect_values(
        _mapping(publication, "release"),
        {
            "deadline": "no_later_than_submission",
            "code": "full_versioned_source_and_dependency_lock",
            "protocol_configs_and_authorizations": True,
            "qualified_checkpoints_and_manifests": True,
            "transition_decision_trace_metric_and_gate_artifacts": True,
            "uncertainty_integrity_failure_and_interruption_artifacts": True,
            "manuscript_table_and_figure_source_data": True,
            "hash_algorithm": "sha256",
            "human_or_personal_data_present": False,
        },
        "publication_evidence.release",
    )
    _expect_values(
        _mapping(publication, "retention"),
        {
            "minimum_years_after_final_disposition": 5,
            "preserve_through_peer_review": True,
            "preserve_failed_invalid_and_interrupted_artifacts": True,
            "silent_deletion_or_replacement_allowed": False,
        },
        "publication_evidence.retention",
    )
    _expect_values(
        _mapping(publication, "compute_reporting"),
        {
            "required_fields": [
                "source_revision",
                "config_identity",
                "authorization_identity",
                "exact_command",
                "python_version",
                "package_lock_identity",
                "os",
                "cpu",
                "ram",
                "gpu",
                "driver",
                "cuda",
                "mujoco_version",
                "gymnasium_version",
                "device",
                "start_timestamp",
                "end_timestamp",
                "wall_clock_seconds",
                "peak_gpu_memory",
                "peak_host_memory",
                "disk_bytes",
                "processed_counts",
                "interruption_history",
            ]
        },
        "publication_evidence.compute_reporting",
    )
    _expect_values(
        _mapping(publication, "manuscript_evidence_table"),
        {
            "required_sections": [
                "checkpoint_qualification",
                "development",
                "confirmation",
                "external_walker_gravity",
                "external_halfcheetah_mass",
                "failures_and_invalidity",
                "compute_and_artifact_identities",
            ],
            "every_condition_and_method_required": True,
            "exact_counts_and_point_estimates_required": True,
            "support_abstention_and_missingness_required": True,
            "checkpoint_cluster_uncertainty_required": True,
            "development_confirmation_pooling_allowed": False,
            "failed_regime_or_method_omission_allowed": False,
        },
        "publication_evidence.manuscript_evidence_table",
    )
    _expect_values(
        _mapping(data, "evidence_boundary"),
        {
            "cotm_changed_streams_allowed_for_socis": False,
            "cotm_calibration_allowed_for_socis": False,
            "synthetic_reachability_is_scientific_evidence": False,
            "manuscript_results_population_allowed": False,
            "policy_reuse_or_ope_allowed": False,
        },
        "evidence_boundary",
    )
    _expect_values(
        _mapping(data, "streams"),
        {
            "warmup_windows": 40,
            "monitoring_windows": 80,
            "window_size": 256,
            "miss_restricted_delay_windows": 81,
        },
        "streams",
    )
    expected_conditions = {
        "stable_frozen": {
            "gravity_multiplier": 1.0,
            "policy_mode": "frozen",
            "updates": False,
            "change_after_warmup": False,
        },
        "stable_updating": {
            "gravity_multiplier": 1.0,
            "policy_mode": "updating",
            "updates": True,
            "change_after_warmup": False,
        },
        "changed_frozen_low": {
            "gravity_multiplier": 0.75,
            "policy_mode": "frozen",
            "updates": False,
            "change_after_warmup": True,
        },
        "changed_frozen_high": {
            "gravity_multiplier": 1.25,
            "policy_mode": "frozen",
            "updates": False,
            "change_after_warmup": True,
        },
        "changed_updating_low": {
            "gravity_multiplier": 0.75,
            "policy_mode": "updating",
            "updates": True,
            "change_after_warmup": True,
        },
        "changed_updating_high": {
            "gravity_multiplier": 1.25,
            "policy_mode": "updating",
            "updates": True,
            "change_after_warmup": True,
        },
    }
    _expect(
        _mapping(_mapping(data, "streams"), "conditions"),
        expected_conditions,
        "streams.conditions",
    )
    _expect_values(
        _mapping(data, "replay_split"),
        {
            "source": "qualified_stable_checkpoint_replay_only",
            "row_identity_fields": [
                "checkpoint_seed",
                "replay_row_index",
                "serialized_transition_sha256",
            ],
            "hash_prefix": "socis-replay-split-v1",
            "ordering": "sha256_ascending",
            "model_train_fraction": 0.60,
            "model_validation_fraction": 0.20,
            "residual_reference_fraction": 0.20,
            "boundary_rounding": (
                "floor_train_then_floor_validation_remainder_reference"
            ),
            "disjoint_required": True,
            "exhaustive_required": True,
            "persist_assignments": True,
            "changed_rows_allowed": False,
            "monitoring_rows_allowed": False,
        },
        "replay_split",
    )
    _expect_values(
        _mapping(data, "preprocessing"),
        {
            "fit_split": "model_train",
            "location": "median",
            "scale": "1.4826_mad",
            "minimum_scale": 1.0e-6,
            "model_input_clip": 10.0,
            "residual_clip": None,
            "dtype": "float64",
        },
        "preprocessing",
    )
    _expect_values(
        _mapping(data, "outcome_model"),
        {
            "architecture": "mlp",
            "hidden_layers": [256, 256],
            "activation": "silu",
            "optimizer": "adam",
            "learning_rate": 3.0e-4,
            "batch_size": 1024,
            "maximum_epochs": 100,
            "patience": 10,
            "validation_tie_break": "earliest_epoch",
            "continuous_loss": "huber",
            "huber_delta": 1.0,
            "continuous_loss_weight": 1.0,
            "binary_loss": "binary_cross_entropy",
            "binary_loss_weight": 1.0,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "fit_changed_data": False,
            "freeze_after_validation": True,
            "fit_seed_namespace": 20260808,
            "fit_seed_derivation": (
                "numpy_seedsequence_namespace_and_fit_code_first_uint32"
            ),
            "loco_fit_code": "held_out_checkpoint_seed",
            "final_fit_code": 0,
            "fit_device_when_authorized": "cuda:0",
            "pooled_replay_order": ("checkpoint_seed_then_physical_replay_row_index"),
        },
        "outcome_model",
    )
    _expect_values(
        _mapping(data, "residuals"),
        {
            "dimension": 14,
            "standardization_split": "residual_reference",
            "location": "median",
            "scale": "1.4826_mad",
            "minimum_scale": 1.0e-6,
            "actor_or_update_features_allowed": False,
        },
        "residuals",
    )
    _expect_values(
        _mapping(data, "support"),
        {
            "distance_space": "standardized_state_action",
            "metric": "euclidean",
            "neighbors": 32,
            "self_query_neighbors": 33,
            "discard_exact_self_match": True,
            "boundary_quantile": 0.99,
            "quantile_method": "higher",
            "matched_neighbor_allocation": "equal",
            "supported_mass_minimum": 0.5,
            "current_effective_sample_size_minimum": 128,
            "reference_effective_sample_size_minimum": 128,
            "failed_state": "ABSTAIN_SUPPORT",
            "hold_sequential_statistic_on_failure": True,
        },
        "support",
    )
    _expect_values(
        _mapping(data, "rff_mmd"),
        {
            "kernel": "gaussian",
            "random_features": 256,
            "base_frequency_distribution": "standard_normal_14d",
            "phase_distribution": "uniform_0_2pi",
            "rng": "numpy_pcg64",
            "rng_seed": 20260803,
            "estimator": "weighted_unbiased_off_diagonal_rff",
            "allow_negative_score": True,
            "truncate_score": False,
            "compute_dtype": "float64",
            "conservative_score_lower_bound": -8.0,
            "conservative_score_upper_bound": 8.0,
        },
        "rff_mmd",
    )
    _expect_values(
        _mapping(_mapping(data, "rff_mmd"), "bandwidth"),
        {
            "source": "residual_reference",
            "statistic": "median_pairwise_distance",
            "quantile_method": "higher",
            "deterministic_hash_sample_rows": 4096,
        },
        "rff_mmd.bandwidth",
    )
    _expect_values(
        _mapping(data, "null_standardization"),
        {
            "modes": ["frozen", "updating"],
            "source_conditions": {
                "frozen": "stable_frozen",
                "updating": "stable_updating",
            },
            "loco_calibration_streams_per_mode": 36,
            "statistic_location": "mean",
            "statistic_scale": "sample_standard_deviation",
            "standard_deviation_ddof": 1,
            "minimum_standard_deviation": 1.0e-12,
            "equal_actor_magnitude_residualization": False,
            "actor_change_subtraction": False,
            "changed_data_allowed": False,
        },
        "null_standardization",
    )
    sequential = _mapping(data, "sequential")
    _expect_values(
        sequential,
        {
            "method": "one_sided_unknown_mean_glr_scan",
            "drift_parameter": None,
            "monitoring_horizon": 80,
            "abstention_behavior": "hold",
            "stopping_comparison": "strict_greater",
        },
        "sequential",
    )
    _expect_values(
        _mapping(sequential, "final_threshold"),
        {
            "stable_streams_per_mode": 40,
            "stable_stream_maximum_quantile": 0.99,
            "quantile_method": "higher",
            "per_mode_then_maximum": True,
            "changed_data_allowed": False,
        },
        "sequential.final_threshold",
    )
    reachability = _mapping(data, "reachability")
    _expect_values(
        _mapping(reachability, "analytic"),
        {
            "required_for_each_mode": True,
            "score_upper_bound": 8.0,
            "reject_when_threshold_greater_or_equal": True,
        },
        "reachability.analytic",
    )
    _expect_values(
        _mapping(reachability, "synthetic"),
        {
            "source": "stable_residual_reference_only",
            "seed": 20260804,
            "blocks": 80,
            "block_size": 256,
            "mean_shift_deltas": [0.25, 0.5, 1.0, 2.0],
            "variance_multipliers": [1.25, 1.5, 2.0],
            "apply_to_all_residual_channels": True,
            "required_mean_crossing_delta": 2.0,
            "required_variance_crossing_multiplier": 2.0,
            "required_in_each_policy_mode": True,
            "observed_changed_streams_allowed": False,
        },
        "reachability.synthetic",
    )
    comparators = _mapping(data, "comparators")
    _expect_values(
        comparators,
        {
            "common_threshold_calibration": "loco_stable_only_0.99_higher",
            "stopping_comparison": "strict_greater",
        },
        "comparators",
    )
    _expect_values(
        _mapping(comparators, "memoryless"),
        {
            "input": "socis_z_score",
            "evidence": "positive_part",
            "accumulation": False,
        },
        "comparators.memoryless",
    )
    _expect_values(
        _mapping(comparators, "adwin"),
        {
            "input": "socis_z_score",
            "native_view": "standard_normal_cdf",
            "delta": 0.1,
            "minimum_subwindow_length": 5,
            "exhaustive_splits": True,
            "inherited_serial_rl_guarantee": False,
        },
        "comparators.adwin",
    )
    watch = _mapping(comparators, "watch")
    _expect_values(
        watch,
        {
            "input_x": "mean_standardized_state_action_centroid_per_block",
            "nonconformity": (
                "mean_squared_l2_standardized_outcome_residual_per_block"
            ),
            "calibration_split": "residual_reference",
            "betting_strategy": "composite_jumper",
            "adaptation_trigger": "x_ctm",
            "empirical_rl_threshold_calibration": ("loco_stable_only_0.99_higher"),
        },
        "comparators.watch",
    )
    _expect_values(
        _mapping(watch, "density_ratio"),
        {
            "model": "logistic_regression",
            "fitting": "causal_reference_plus_strictly_prior_monitoring_blocks",
            "include_current_block_in_fit": False,
            "class_weight": None,
            "solver": "lbfgs",
            "c": 1.0,
            "maximum_iterations": 1000,
            "tolerance": 1.0e-10,
            "random_state": 20260806,
            "probability_clip": 1.0e-12,
        },
        "comparators.watch.density_ratio",
    )
    _expect_values(
        _mapping(data, "gates"),
        {
            "stable_eligible_block_fraction_minimum_per_mode": 0.90,
            "changed_eligible_block_fraction_minimum_per_regime": 0.80,
            "stable_false_alarms_maximum_per_mode": 2,
            "stable_streams_per_mode": 40,
            "changed_minimum_detections_per_regime": 1,
            "changed_streams_per_regime": 40,
            "changed_updating_low_minimum_detections": 24,
            "changed_updating_low_streams": 40,
            "memoryless_relative_mean_restricted_delay_improvement_minimum": 0.30,
            "freeze_only_when_all_pass": True,
        },
        "gates",
    )
    _expect_values(
        _mapping(data, "statistics"),
        {
            "inferential_cluster": "checkpoint_seed",
            "within_development_seed_replicates": 4,
            "confidence": 0.95,
            "whole_cluster_bootstrap_draws": 50_000,
            "cluster_bootstrap_seed": 20260805,
            "fixed_horizon_false_alarm_definition": (
                "any_strict_crossing_in_80_monitoring_windows"
            ),
            "detection_delay_origin": "first_monitoring_window",
            "miss_restricted_delay": 81,
            "gate_uses": "exact_counts_and_point_estimates",
            "intervals_are_population_guarantees": False,
        },
        "statistics",
    )
    schemas = _mapping(data, "schemas")
    _expect_values(
        _mapping(schemas, "transition_record"),
        {
            "observation_dimension": 11,
            "action_dimension": 3,
            "continuous_dtype": "float64",
            "parquet_compression": "zstd",
            "required_audit_fields": [
                "policy_version",
                "actor_update_count",
                "critic_update_count",
                "gravity_multiplier",
            ],
        },
        "schemas.transition_record",
    )
    _expect_values(
        _mapping(schemas, "block_decision"),
        {
            "states": [
                "WARMUP",
                "MONITORING",
                "ABSTAIN_SUPPORT",
                "ALARM",
                "POST_ALARM",
            ],
            "required_support_fields": [
                "supported_mass",
                "current_ess",
                "reference_ess",
                "support_boundary",
            ],
            "required_score_fields": [
                "mmd_u_rff",
                "null_mean",
                "null_std",
                "z_score",
                "glr_value",
            ],
        },
        "schemas.block_decision",
    )
    _expect_values(
        _mapping(data, "artifacts"),
        {
            "atomic_stream_shards": True,
            "transition_shard_per_stream": True,
            "decision_shard_per_stream": True,
            "preserve_failed_shards": True,
            "resume_complete_shards": True,
            "reject_symlinked_or_unknown_content": True,
            "fsync_before_atomic_rename": True,
            "hash_algorithm": "sha256",
            "freeze_filename": "socis_freeze_v1.json",
        },
        "artifacts",
    )
    expected_transitions = [
        ["PREREGISTERED", "CHECKPOINTS_VERIFIED"],
        ["CHECKPOINTS_VERIFIED", "REPLAY_SPLITS_LOCKED"],
        ["REPLAY_SPLITS_LOCKED", "STABLE_STREAMS_COMPLETE"],
        ["STABLE_STREAMS_COMPLETE", "CALIBRATION_LOCKED"],
        ["CALIBRATION_LOCKED", "CHANGED_STREAMS_COMPLETE"],
        ["CHANGED_STREAMS_COMPLETE", "DEVELOPMENT_ANALYZED"],
        ["DEVELOPMENT_ANALYZED", "DEVELOPMENT_FAILED"],
        ["DEVELOPMENT_ANALYZED", "FREEZE_WRITTEN"],
        ["CHECKPOINTS_VERIFIED", "DEVELOPMENT_FAILED"],
        ["REPLAY_SPLITS_LOCKED", "DEVELOPMENT_FAILED"],
        ["STABLE_STREAMS_COMPLETE", "DEVELOPMENT_FAILED"],
        ["CALIBRATION_LOCKED", "DEVELOPMENT_FAILED"],
        ["CHANGED_STREAMS_COMPLETE", "DEVELOPMENT_FAILED"],
    ]
    _expect_values(
        _mapping(data, "run_state"),
        {
            "transitions": expected_transitions,
            "scientific_failure_terminal_from_completed_stage": True,
            "operational_interruption_behavior": (
                "retain_last_verified_state_and_resume"
            ),
            "confirmation_transition_in_develop_command": False,
        },
        "run_state",
    )
    _expect_values(
        _mapping(data, "execution"),
        {
            "implementation_present": True,
            "gpu_execution_authorized": False,
            "checkpoint_training_authorized": False,
            "checkpoint_training_authorization_source": (
                "external_self_hashed_phase_c_record"
            ),
            "device_when_authorized": "cuda:0",
            "scoring_dtype": "float64",
            "heartbeat_seconds": 60,
            "minimum_free_gpu_memory_mib": 8192,
            "minimum_free_disk_gib": 20,
            "allowlisted_non_scientific_gpu_processes": [
                {
                    "executable": "/usr/libexec/gnome-remote-desktop-daemon",
                    "maximum_used_memory_mib": 1024,
                }
            ],
        },
        "execution",
    )


def validate_protocol(data: dict[str, Any]) -> None:
    """Validate identities, arithmetic, seed walls, and execution guards."""
    protocol = _mapping(data, "protocol")
    _expect(protocol.get("name"), "socis_rl", "protocol.name")
    _expect(protocol.get("version"), 1, "protocol.version")
    _expect(protocol.get("amendment"), 2, "protocol.amendment")
    _expect(protocol.get("operational_revision"), 3, "protocol.operational_revision")
    _expect(
        protocol.get("status"),
        "implemented_not_executed",
        "protocol.status",
    )
    recorded = protocol.get("config_identity_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise SOCISProtocolViolation("configuration identity must be SHA-256")
    computed = zero_field_identity(data, "protocol.config_identity_sha256")
    if recorded != computed:
        raise SOCISProtocolViolation("configuration self-hash mismatch")

    seeds = _mapping(data, "seeds")
    development_seeds = _integers(seeds, "development_checkpoint_seeds")
    confirmation_seeds = _integers(seeds, "confirmation_checkpoint_seeds")
    forbidden_seeds = _integers(seeds, "forbidden_legacy_checkpoint_seeds")
    _expect(development_seeds, list(DEVELOPMENT_SEEDS), "development seeds")
    _expect(confirmation_seeds, list(CONFIRMATION_SEEDS), "confirmation seeds")
    _expect(forbidden_seeds, list(FORBIDDEN_LEGACY_SEEDS), "legacy seeds")
    if set(development_seeds) & (set(confirmation_seeds) | set(forbidden_seeds)):
        raise SOCISProtocolViolation("development seed sets are not disjoint")
    if set(confirmation_seeds) & set(forbidden_seeds):
        raise SOCISProtocolViolation("confirmation and legacy seed sets overlap")

    development = _mapping(data, "development")
    _expect(
        _integers(development, "checkpoint_seeds"),
        list(DEVELOPMENT_SEEDS),
        "development.checkpoint_seeds",
    )
    replicates = _integers(development, "replicates")
    _expect(replicates, [0, 1, 2, 3], "development.replicates")
    conditions = development.get("conditions")
    _expect(conditions, list(CONDITIONS), "development.conditions")

    streams = _mapping(data, "streams")
    stream_conditions = streams.get("conditions")
    if (
        not isinstance(stream_conditions, dict)
        or tuple(stream_conditions) != CONDITIONS
    ):
        raise SOCISProtocolViolation("streams.conditions must be the ordered regimes")
    total_windows = int(streams.get("warmup_windows", -1)) + int(
        streams.get("monitoring_windows", -1)
    )
    window_size = int(streams.get("window_size", -1))
    expected_streams = len(DEVELOPMENT_SEEDS) * len(replicates) * len(CONDITIONS)
    expected_windows = expected_streams * total_windows
    expected_transitions = expected_windows * window_size
    for key, expected in (
        ("expected_streams", expected_streams),
        ("expected_windows", expected_windows),
        ("expected_transitions", expected_transitions),
    ):
        _expect(development.get(key), expected, f"development.{key}")
    _expect(development.get("stable_streams_per_mode"), 40, "stable count")
    _expect(
        development.get("loco_stable_calibration_streams_per_mode"),
        36,
        "LOCO stable count",
    )

    confirmation = _mapping(data, "confirmation")
    _expect(confirmation.get("enabled"), False, "confirmation.enabled")
    _expect(
        _integers(confirmation, "checkpoint_seeds"),
        list(CONFIRMATION_SEEDS),
        "confirmation.checkpoint_seeds",
    )
    confirmation_replicates = _integers(confirmation, "replicates")
    confirmation_conditions = confirmation.get("conditions")
    _expect(confirmation_conditions, list(CONDITIONS), "confirmation.conditions")
    confirmation_streams = (
        len(CONFIRMATION_SEEDS) * len(confirmation_replicates) * len(CONDITIONS)
    )
    for key, expected in (
        ("expected_streams", confirmation_streams),
        ("expected_windows", confirmation_streams * total_windows),
        (
            "expected_transitions",
            confirmation_streams * total_windows * window_size,
        ),
    ):
        _expect(confirmation.get(key), expected, f"confirmation.{key}")

    target = _mapping(data, "target")
    _expect(
        target.get("actor_change_residualization_allowed"),
        False,
        "actor residualization",
    )
    checkpoint = _mapping(data, "checkpoint_contract")
    _expect(
        checkpoint.get("checkpoint_manifest_filename"),
        "socis_checkpoint_manifest_v2.json",
        "checkpoint manifest filename",
    )
    _expect(
        checkpoint.get("checkpoint_manifest_schema_version"),
        2,
        "checkpoint manifest schema version",
    )
    _expect(
        checkpoint.get("checkpoint_creation_requires_separate_review"),
        True,
        "checkpoint creation review",
    )

    watch = _mapping(_mapping(data, "comparators"), "watch")
    locked_watch = {
        "source_revision": WATCH_REVISION,
        "implementation_claim": "prespecified_blockwise_variant_not_paper_identical",
        "block_size": 256,
        "output_granularity": "one_wealth_per_256_transition_block",
        "final_stopping_comparison": "strict_greater",
        "inherited_anytime_valid_claim": False,
    }
    for key, locked_value in locked_watch.items():
        _expect(watch.get(key), locked_value, f"comparators.watch.{key}")
    rank = _mapping(watch, "conformal_rank")
    _expect(rank.get("tie_break_seed"), 20260806, "WATCH tie seed")
    _expect(rank.get("comparison"), "strict_greater", "WATCH rank comparison")
    jumper = _mapping(watch, "composite_jumper")
    _expect(
        jumper.get("jump_rates"),
        [0.0001, 0.001, 0.01, 0.1, 1.0],
        "WATCH jump rates",
    )
    _expect(
        _mapping(watch, "x_ctm").get("strict_trigger_wealth"),
        100.0,
        "WATCH X-CTM trigger",
    )
    ratio = _mapping(watch, "density_ratio")
    _expect(ratio.get("include_current_block_in_fit"), False, "WATCH causal fitting")

    execution = _mapping(data, "execution")
    _expect(execution.get("implementation_present"), True, "implementation")
    _expect(execution.get("gpu_execution_authorized"), False, "GPU authorization")
    _expect(
        execution.get("checkpoint_training_authorized"),
        False,
        "checkpoint training authorization",
    )
    run_state = _mapping(data, "run_state")
    _expect(
        run_state.get("scientific_failure_terminal_from_completed_stage"),
        True,
        "scientific terminal failure",
    )
    _expect(
        run_state.get("operational_interruption_behavior"),
        "retain_last_verified_state_and_resume",
        "operational resume behavior",
    )
    _expect(confirmation_replicates, [0, 1, 2, 3, 4], "confirmation.replicates")
    _expect(
        confirmation.get("confirmation_data_allowed_for_refit_or_calibration"),
        False,
        "confirmation refit or calibration",
    )
    _expect(confirmation.get("retuning_after_outcome"), False, "confirmation retuning")
    _validate_protocol_header(data)
    _validate_confirmation_analysis(confirmation)
    _validate_phase_c_operational(data)
    _validate_external_and_target(data)
    _validate_publication_and_science(data)


@dataclass(frozen=True)
class SOCISConfig:
    """Validated configuration and its canonical identity."""

    path: Path
    data: dict[str, Any]
    sha256: str
    raw_sha256: str

    def section(self, name: str) -> dict[str, Any]:
        return _mapping(self.data, name)

    @property
    def workspace(self) -> Path:
        for parent in self.path.parents:
            if (parent / "pyproject.toml").is_file() and (
                parent / "src" / "rl_hanbat"
            ).is_dir():
                return parent
        raise SOCISProtocolViolation(f"cannot locate workspace above {self.path}")

    @property
    def development_seeds(self) -> tuple[int, ...]:
        return tuple(
            int(value) for value in self.section("development")["checkpoint_seeds"]
        )


def load_config(path: str | Path, *, verify_proposal: bool = True) -> SOCISConfig:
    """Load the locked config and optionally verify its proposal ancestor."""
    config_path = Path(path).expanduser().resolve()
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SOCISProtocolViolation("configuration root must be a mapping")
    validate_protocol(loaded)
    identity = zero_field_identity(loaded, "protocol.config_identity_sha256")
    config = SOCISConfig(config_path, loaded, identity, file_sha256(config_path))
    if verify_proposal:
        proposal = config.workspace / "experiments" / "socis_rl" / "METHOD_PROPOSAL.md"
        expected = str(config.section("protocol")["method_proposal_sha256"])
        if file_sha256(proposal) != expected:
            raise SOCISProtocolViolation("method proposal hash mismatch")
        amendment_1 = (
            config.workspace / "experiments" / "socis_rl" / "PROTOCOL_AMENDMENT_1.md"
        )
        if file_sha256(amendment_1) != str(
            config.section("protocol")["amendment_1_sha256"]
        ):
            raise SOCISProtocolViolation("protocol amendment 1 hash mismatch")
        amendment_2 = (
            config.workspace / "experiments" / "socis_rl" / "PROTOCOL_AMENDMENT_2.md"
        )
        if file_sha256(amendment_2) != str(
            config.section("protocol")["amendment_sha256"]
        ):
            raise SOCISProtocolViolation("protocol amendment 2 hash mismatch")
        operational = (
            config.workspace
            / "experiments"
            / "socis_rl"
            / "PHASE_C_OPERATIONAL_LOCK.md"
        )
        expected_operational = str(
            config.section("protocol")["phase_c_operational_lock_sha256"]
        )
        if file_sha256(operational) != expected_operational:
            raise SOCISProtocolViolation("Phase C operational lock hash mismatch")
        phase_d_operational = (
            config.workspace
            / "experiments"
            / "socis_rl"
            / "PHASE_D_OPERATIONAL_LOCK.md"
        )
        if file_sha256(phase_d_operational) != str(
            config.section("protocol")["phase_d_operational_lock_sha256"]
        ):
            raise SOCISProtocolViolation("Phase D operational lock hash mismatch")
    return config
