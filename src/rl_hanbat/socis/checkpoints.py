"""Dedicated seed-30--39 checkpoint manifest verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from rl_hanbat.identification.utils import deterministic_seed, model_component_hashes
from rl_hanbat.socis.authorization import AUTHORIZATION_FIELDS
from rl_hanbat.socis.config import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    FORBIDDEN_LEGACY_SEEDS,
    SOCISConfig,
)
from rl_hanbat.socis.hashing import (
    canonical_sha256,
    file_sha256,
    source_revision,
    source_tree_sha256,
    zero_field_identity,
)
from rl_hanbat.socis.persistence import ArtifactIntegrityError, require_regular_file
from rl_hanbat.socis.schema import arrow_schema_identity, checkpoint_replay_schema

_FORBIDDEN_SEED_TOKEN = re.compile(
    r"(?i)(?:checkpoint[_ -]?seed|seed)[_ :=/-]*(\d{1,3})"
)
_MANIFEST_FIELDS = {
    "schema_version",
    "purpose",
    "config_sha256",
    "config_raw_sha256",
    "method_proposal_sha256",
    "amendment_sha256",
    "phase_c_operational_lock_sha256",
    "source_revision",
    "source_tree_sha256",
    "authorization_sha256",
    "authorization_file_sha256",
    "preflight_file_sha256",
    "training_contract_sha256",
    "checkpoint_seeds",
    "checkpoints",
}
_RECORD_FIELDS = {
    "checkpoint_seed",
    "training_seed",
    "evaluation_seed",
    "probe_seed",
    "selected_step",
    "qualification_attempts",
    "median_return",
    "checkpoint_sha256",
    "files",
    "retained_stage",
    "stable_replay",
    "progress",
    "component_hashes",
}
_CHECKPOINT_FILENAMES = {
    "model": "model.zip",
    "replay_buffer": "replay_buffer.pkl",
    "rng_state": "rng_state.pt",
    "probe_states": "probe_states.npy",
}


@dataclass(frozen=True)
class QualifiedCheckpoint:
    """Explicit verified paths and identities for one development checkpoint."""

    seed: int
    checkpoint_sha256: str
    replay_path: Path
    replay_sha256: str
    replay_rows: int
    files: dict[str, Path]
    file_hashes: dict[str, str]
    component_hashes: dict[str, str]


@dataclass(frozen=True)
class CheckpointPanel:
    """The exact verified seed-30--39 checkpoint panel."""

    manifest_path: Path
    manifest_sha256: str
    checkpoints: dict[int, QualifiedCheckpoint]


def _reject_forbidden_mentions(value: Any, *, key_hint: str = "") -> None:
    forbidden = set(FORBIDDEN_LEGACY_SEEDS) | set(CONFIRMATION_SEEDS)
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_forbidden_mentions(child, key_hint=str(key))
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_mentions(child, key_hint=key_hint)
    elif isinstance(value, int) and "seed" in key_hint.lower() and value in forbidden:
        raise ArtifactIntegrityError(f"manifest mentions forbidden seed {value}")
    elif isinstance(value, str):
        for match in _FORBIDDEN_SEED_TOKEN.finditer(value):
            seed = int(match.group(1))
            if seed in forbidden:
                raise ArtifactIntegrityError(f"manifest mentions forbidden seed {seed}")


def _explicit_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ArtifactIntegrityError("checkpoint path must be a nonempty string")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ArtifactIntegrityError(f"checkpoint path escapes source root: {relative}")
    root_resolved = root.resolve()
    candidate = root / candidate_relative
    current = root
    for part in candidate_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactIntegrityError(f"checkpoint path contains symlink: {current}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ArtifactIntegrityError(f"checkpoint path escapes source root: {relative}")
    require_regular_file(candidate)
    return candidate


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ArtifactIntegrityError(f"{name} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ArtifactIntegrityError(f"{name} must be hexadecimal") from error
    return value


def _resource_environment_identity(observation: dict[str, Any]) -> str:
    stable = json.loads(json.dumps(observation, allow_nan=False))
    stable.pop("recorded_at_utc", None)
    stable.pop("disk_free_bytes", None)
    cuda = stable.get("cuda")
    if isinstance(cuda, dict):
        cuda.pop("free_memory_bytes", None)
        cuda.pop("active_compute_processes", None)
    return canonical_sha256(stable)


def _descriptor(
    root: Path,
    seed_directory: Path,
    value: Any,
    name: str,
) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ArtifactIntegrityError(f"{name} descriptor differs")
    path = _explicit_path(root, value.get("path"))
    if path.parent != seed_directory and not path.parent.is_relative_to(seed_directory):
        raise ArtifactIntegrityError(f"{name} path differs from seed directory")
    expected = _sha(value.get("sha256"), name)
    if file_sha256(path) != expected:
        raise ArtifactIntegrityError(f"{name} hash mismatch")
    return path, expected


def _validate_attempts(config: SOCISConfig, seed: int, record: dict[str, Any]) -> None:
    qualification = config.section("checkpoint_training")["qualification"]
    initial = int(qualification["initial_evaluation_steps"])
    maximum = int(qualification["maximum_extension_steps"])
    selected = record.get("selected_step")
    if selected not in {initial, maximum}:
        raise ArtifactIntegrityError(f"seed {seed} selected step differs")
    expected_steps = [initial] if selected == initial else [initial, maximum]
    attempts = record.get("qualification_attempts")
    if not isinstance(attempts, list) or len(attempts) != len(expected_steps):
        raise ArtifactIntegrityError(f"seed {seed} qualification attempts differ")
    threshold = float(qualification["minimum_median_return"])
    episode_count = int(qualification["evaluation_episodes"])
    for index, (attempt, step) in enumerate(zip(attempts, expected_steps, strict=True)):
        if not isinstance(attempt, dict) or set(attempt) != {
            "checkpoint_seed",
            "step",
            "evaluation_returns",
            "median_return",
            "finite_trajectory",
            "qualified",
        }:
            raise ArtifactIntegrityError(f"seed {seed} qualification schema differs")
        returns = attempt.get("evaluation_returns")
        if (
            attempt.get("checkpoint_seed") != seed
            or attempt.get("step") != step
            or not isinstance(returns, list)
            or len(returns) != episode_count
        ):
            raise ArtifactIntegrityError(f"seed {seed} qualification values differ")
        values = np.asarray(returns, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ArtifactIntegrityError(f"seed {seed} returns are nonfinite")
        median = float(np.median(values))
        qualified = median >= threshold
        if (
            attempt.get("finite_trajectory") is not True
            or float(attempt.get("median_return", float("nan"))) != median
            or attempt.get("qualified") is not qualified
            or (index < len(attempts) - 1 and qualified)
        ):
            raise ArtifactIntegrityError(f"seed {seed} qualification differs")
    if attempts[-1]["qualified"] is not True or float(
        record.get("median_return", float("nan"))
    ) != float(attempts[-1]["median_return"]):
        raise ArtifactIntegrityError(f"seed {seed} selected qualification differs")


def _validate_replay(
    config: SOCISConfig,
    path: Path,
    seed: int,
    replay: dict[str, Any],
    record_step: int,
    *,
    expected_config_sha256: str | None = None,
) -> int:
    table = pq.read_table(path)
    schema = checkpoint_replay_schema(config)
    rows = replay.get("rows")
    expected_rows = min(
        int(record_step),
        int(config.section("checkpoint_training")["replay_capacity"]),
    )
    if (
        table.schema.remove_metadata() != schema
        or not isinstance(rows, int)
        or rows != expected_rows
        or len(table) != rows
        or replay.get("schema_sha256") != arrow_schema_identity(schema)
    ):
        raise ArtifactIntegrityError(f"seed {seed} stable replay schema differs")
    metadata = table.schema.metadata or {}
    expected_metadata = {
        "schema_version": "1",
        "config_sha256": (
            config.sha256 if expected_config_sha256 is None else expected_config_sha256
        ),
        "checkpoint_seed": str(seed),
        "condition": "stable_frozen",
    }
    for key, value in expected_metadata.items():
        if metadata.get(key.encode()) != value.encode():
            raise ArtifactIntegrityError(f"seed {seed} replay metadata differs")
    if table["checkpoint_seed"].to_pylist() != [seed] * rows:
        raise ArtifactIntegrityError(f"seed {seed} replay seed differs")
    if table["replay_row_index"].to_pylist() != list(range(rows)):
        raise ArtifactIntegrityError(f"seed {seed} replay order differs")
    if table["condition"].to_pylist() != ["stable_frozen"] * rows:
        raise ArtifactIntegrityError(f"seed {seed} replay condition differs")
    for name in ("observation", "action", "next_observation", "reward"):
        array = table[name].combine_chunks()
        values = (
            array.values.to_numpy(zero_copy_only=False)
            if pa.types.is_fixed_size_list(array.type)
            else array.to_numpy(zero_copy_only=False)
        )
        if not np.isfinite(values).all():
            raise ArtifactIntegrityError(f"seed {seed} replay is nonfinite")
    return rows


def _verify_manifest_path(
    config: SOCISConfig,
    root: Path,
    manifest_path: Path,
    *,
    expected_authorization_sha256: str | None,
    frozen_manifest_sha256: str | None = None,
) -> CheckpointPanel:
    """Verify one explicit v2 manifest path and every bound artifact."""
    require_regular_file(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    if frozen_manifest_sha256 is not None and manifest_sha256 != frozen_manifest_sha256:
        raise ArtifactIntegrityError("frozen checkpoint manifest hash differs")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("checkpoint manifest is not valid JSON") from error
    if not isinstance(data, dict) or set(data) != _MANIFEST_FIELDS:
        raise ArtifactIntegrityError("checkpoint manifest schema differs")
    _reject_forbidden_mentions(data)
    contract = config.section("checkpoint_contract")
    protocol = config.section("protocol")
    if frozen_manifest_sha256 is None:
        revision = source_revision(config.workspace)
        ancestry = {
            "schema_version": int(contract["checkpoint_manifest_schema_version"]),
            "purpose": "socis_development_checkpoints",
            "config_sha256": config.sha256,
            "config_raw_sha256": config.raw_sha256,
            "method_proposal_sha256": protocol["method_proposal_sha256"],
            "amendment_sha256": protocol["amendment_sha256"],
            "phase_c_operational_lock_sha256": protocol[
                "phase_c_operational_lock_sha256"
            ],
            "source_revision": revision,
            "source_tree_sha256": source_tree_sha256(config.workspace),
            "training_contract_sha256": canonical_sha256(
                config.section("checkpoint_training")
            ),
        }
        if revision is None or any(
            data.get(key) != value for key, value in ancestry.items()
        ):
            raise ArtifactIntegrityError("checkpoint manifest ancestry differs")
    else:
        # Gate 1 is immutable input to later executable source. Its raw
        # manifest hash binds the historic config/source identities; comparing
        # them to the evolved Phase D checkout would erase the provenance
        # boundary instead of strengthening it.
        ancestry = {
            name: data[name]
            for name in (
                "schema_version",
                "purpose",
                "config_sha256",
                "config_raw_sha256",
                "method_proposal_sha256",
                "amendment_sha256",
                "phase_c_operational_lock_sha256",
                "source_revision",
                "source_tree_sha256",
                "training_contract_sha256",
            )
        }
        revision = ancestry["source_revision"]
        if (
            ancestry["schema_version"]
            != int(contract["checkpoint_manifest_schema_version"])
            or ancestry["purpose"] != "socis_development_checkpoints"
            or ancestry["method_proposal_sha256"] != protocol["method_proposal_sha256"]
            or ancestry["training_contract_sha256"]
            != canonical_sha256(config.section("checkpoint_training"))
            or not isinstance(revision, str)
            or len(revision) != 40
        ):
            raise ArtifactIntegrityError("frozen checkpoint ancestry differs")
    authorization_sha = _sha(
        data.get("authorization_sha256"), "checkpoint authorization"
    )
    if (
        expected_authorization_sha256 is not None
        and authorization_sha != expected_authorization_sha256
    ):
        raise ArtifactIntegrityError("checkpoint authorization identity differs")
    authorization_path = root / "authorization.json"
    require_regular_file(authorization_path)
    authorization_file_sha = _sha(
        data.get("authorization_file_sha256"), "authorization file"
    )
    if file_sha256(authorization_path) != authorization_file_sha:
        raise ArtifactIntegrityError("checkpoint authorization file hash mismatch")
    try:
        authorization_data = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("checkpoint authorization copy differs") from error
    if (
        not isinstance(authorization_data, dict)
        or set(authorization_data) != AUTHORIZATION_FIELDS
        or authorization_data.get("self_sha256") != authorization_sha
        or zero_field_identity(authorization_data, "self_sha256") != authorization_sha
    ):
        raise ArtifactIntegrityError("checkpoint authorization copy differs")
    authorization_expected = {
        "schema_version": config.section("phase_c_authorization")["schema_version"],
        "purpose": "socis_phase_c_checkpoint_training",
        "decision": "AUTHORIZE_PHASE_C",
        "protocol_name": protocol["name"],
        "protocol_version": protocol["version"],
        "protocol_amendment": (
            protocol["amendment"]
            if frozen_manifest_sha256 is None
            else authorization_data.get("protocol_amendment")
        ),
        "operational_revision": (
            protocol["operational_revision"]
            if frozen_manifest_sha256 is None
            else authorization_data.get("operational_revision")
        ),
        "config_sha256": ancestry["config_sha256"],
        "config_raw_sha256": ancestry["config_raw_sha256"],
        "method_proposal_sha256": ancestry["method_proposal_sha256"],
        "amendment_sha256": ancestry["amendment_sha256"],
        "phase_c_operational_lock_sha256": ancestry["phase_c_operational_lock_sha256"],
        "source_revision": revision,
        "source_tree_sha256": ancestry["source_tree_sha256"],
        "workspace": str(config.workspace.resolve()),
        "config_path": str(config.path),
        "output_root": str(root),
        "checkpoint_seeds": list(DEVELOPMENT_SEEDS),
        "device": "cuda:0",
        "canonical_command": "python -m rl_hanbat.socis train-checkpoints",
        "authorization_scope": "checkpoint_creation_and_verification_only",
        "development_collection_authorized": False,
        "confirmation_authorized": False,
    }
    if any(
        authorization_data.get(name) != value
        for name, value in authorization_expected.items()
    ):
        raise ArtifactIntegrityError("checkpoint authorization ancestry differs")
    preflight_path = root / "phase_c_preflight.json"
    require_regular_file(preflight_path)
    preflight_file_sha = _sha(
        data.get("preflight_file_sha256"), "Phase C preflight file"
    )
    if file_sha256(preflight_path) != preflight_file_sha:
        raise ArtifactIntegrityError("Phase C preflight file hash mismatch")
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("Phase C preflight record differs") from error
    preflight_fields = {
        "schema_version",
        "config_sha256",
        "config_raw_sha256",
        "authorization_sha256",
        "source_revision",
        "source_tree_sha256",
        "workspace",
        "output_root",
        "device",
        "canonical_command",
        "checkpoint_seeds",
        "environment_id",
        "uv_lock_sha256",
        "resource_observation",
        "resource_environment_sha256",
        "self_sha256",
    }
    lock_path = config.workspace / "uv.lock"
    require_regular_file(lock_path)
    preflight_expected = {
        "schema_version": 1,
        "config_sha256": ancestry["config_sha256"],
        "config_raw_sha256": ancestry["config_raw_sha256"],
        "authorization_sha256": authorization_sha,
        "source_revision": revision,
        "source_tree_sha256": ancestry["source_tree_sha256"],
        "workspace": str(config.workspace.resolve()),
        "output_root": str(root),
        "device": "cuda:0",
        "canonical_command": "python -m rl_hanbat.socis train-checkpoints",
        "checkpoint_seeds": list(DEVELOPMENT_SEEDS),
        "environment_id": "Hopper-v5",
        "uv_lock_sha256": file_sha256(lock_path),
    }
    if (
        not isinstance(preflight, dict)
        or set(preflight) != preflight_fields
        or any(
            preflight.get(name) != value for name, value in preflight_expected.items()
        )
        or not isinstance(preflight.get("resource_observation"), dict)
        or preflight.get("resource_environment_sha256")
        != _resource_environment_identity(preflight["resource_observation"])
        or preflight.get("self_sha256") != zero_field_identity(preflight, "self_sha256")
    ):
        raise ArtifactIntegrityError("Phase C preflight record differs")
    if data.get("checkpoint_seeds") != list(DEVELOPMENT_SEEDS):
        raise ArtifactIntegrityError("checkpoint manifest must contain exactly 30--39")
    records = data.get("checkpoints")
    if not isinstance(records, list) or len(records) != len(DEVELOPMENT_SEEDS):
        raise ArtifactIntegrityError("checkpoint manifest record count differs")
    expected_names = [str(value) for value in contract["required_checkpoint_files"]]
    expected_components = [
        str(value) for value in contract["required_component_hashes"]
    ]
    seed_codes = config.section("checkpoint_training")["seed_codes"]
    retained_step = int(
        config.section("checkpoint_training")["qualification"]["retained_stage_steps"]
    )
    panel: dict[int, QualifiedCheckpoint] = {}
    for expected_seed, record in zip(DEVELOPMENT_SEEDS, records, strict=True):
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise ArtifactIntegrityError("checkpoint record schema differs")
        seed = record.get("checkpoint_seed")
        if seed != expected_seed:
            raise ArtifactIntegrityError("checkpoint record ordering differs")
        seed_directory = root / "seeds" / f"seed_{seed}"
        if seed_directory.is_symlink() or not seed_directory.is_dir():
            raise ArtifactIntegrityError(f"seed {seed} directory differs")
        expected_seed_values = {
            "training_seed": deterministic_seed(seed, int(seed_codes["training"])),
            "evaluation_seed": deterministic_seed(seed, int(seed_codes["evaluation"])),
            "probe_seed": deterministic_seed(seed, int(seed_codes["probe_states"])),
        }
        if any(record.get(key) != value for key, value in expected_seed_values.items()):
            raise ArtifactIntegrityError(f"seed {seed} namespace differs")
        if record.get("checkpoint_sha256") != zero_field_identity(
            record, "checkpoint_sha256"
        ):
            raise ArtifactIntegrityError(f"seed {seed} checkpoint identity differs")
        _validate_attempts(config, seed, record)
        files_data = record.get("files")
        if not isinstance(files_data, dict) or set(files_data) != set(expected_names):
            raise ArtifactIntegrityError(f"seed {seed} checkpoint file set differs")
        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for name in expected_names:
            paths[name], hashes[name] = _descriptor(
                root, seed_directory, files_data[name], f"seed {seed} {name}"
            )
            if paths[name] != seed_directory / _CHECKPOINT_FILENAMES[name]:
                raise ArtifactIntegrityError(f"seed {seed} {name} path differs")
        probe = np.load(paths["probe_states"], allow_pickle=False)
        probe_count = int(
            config.section("checkpoint_training")["qualification"]["probe_states"]
        )
        if probe.shape != (probe_count, 11) or not np.isfinite(probe).all():
            raise ArtifactIntegrityError(f"seed {seed} probe states differ")
        retained = record.get("retained_stage")
        if (
            not isinstance(retained, dict)
            or set(retained) != {"step", "files"}
            or retained.get("step") != retained_step
            or not isinstance(retained.get("files"), dict)
            or set(retained["files"]) != {"model", "replay_buffer", "rng_state"}
        ):
            raise ArtifactIntegrityError(f"seed {seed} retained stage differs")
        for name, descriptor in retained["files"].items():
            retained_path, _ = _descriptor(
                root,
                seed_directory,
                descriptor,
                f"seed {seed} retained {name}",
            )
            expected_retained = (
                seed_directory
                / f"retained_stage_{retained_step}"
                / _CHECKPOINT_FILENAMES[name]
            )
            if retained_path != expected_retained:
                raise ArtifactIntegrityError(
                    f"seed {seed} retained {name} path differs"
                )
        replay = record.get("stable_replay")
        if not isinstance(replay, dict) or set(replay) != {
            "path",
            "sha256",
            "rows",
            "schema_sha256",
        }:
            raise ArtifactIntegrityError(f"seed {seed} stable replay differs")
        replay_path, replay_hash = _descriptor(
            root,
            seed_directory,
            {"path": replay["path"], "sha256": replay["sha256"]},
            f"seed {seed} stable replay",
        )
        if replay_path != seed_directory / "stable_replay.parquet":
            raise ArtifactIntegrityError(f"seed {seed} stable replay path differs")
        replay_rows = _validate_replay(
            config,
            replay_path,
            seed,
            replay,
            int(record["selected_step"]),
            expected_config_sha256=str(ancestry["config_sha256"]),
        )
        progress_path, _ = _descriptor(
            root, seed_directory, record.get("progress"), f"seed {seed} progress"
        )
        if progress_path != seed_directory / "training_progress.json":
            raise ArtifactIntegrityError(f"seed {seed} progress path differs")
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(f"seed {seed} progress differs") from error
        if progress != {
            "schema_version": 1,
            "checkpoint_seed": seed,
            "processed_environment_steps": record["selected_step"],
            "current_target_steps": record["selected_step"],
            "status": "qualified",
        }:
            raise ArtifactIntegrityError(f"seed {seed} progress differs")
        component_hashes = record.get("component_hashes")
        if not isinstance(component_hashes, dict) or set(component_hashes) != set(
            expected_components
        ):
            raise ArtifactIntegrityError(f"seed {seed} component hashes differ")
        checked_components = {
            name: _sha(component_hashes[name], f"seed {seed} {name}")
            for name in expected_components
        }
        allowed_top = {
            *_CHECKPOINT_FILENAMES.values(),
            "stable_replay.parquet",
            "training_progress.json",
            "integrity.json",
            f"retained_stage_{retained_step}",
        }
        top_entries = list(seed_directory.iterdir())
        if {entry.name for entry in top_entries} != allowed_top or any(
            entry.is_symlink() for entry in top_entries
        ):
            raise ArtifactIntegrityError(f"seed {seed} content differs")
        retained_directory = seed_directory / f"retained_stage_{retained_step}"
        retained_entries = list(retained_directory.iterdir())
        expected_retained_names = {
            _CHECKPOINT_FILENAMES[name]
            for name in ("model", "replay_buffer", "rng_state")
        }
        if {entry.name for entry in retained_entries} != expected_retained_names or any(
            entry.is_symlink() or not entry.is_file() for entry in retained_entries
        ):
            raise ArtifactIntegrityError(f"seed {seed} retained content differs")
        integrity_path = seed_directory / "integrity.json"
        try:
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(
                f"seed {seed} integrity record differs"
            ) from error
        integrity_expected = {
            "schema_version": 2,
            "config_sha256": ancestry["config_sha256"],
            "config_raw_sha256": ancestry["config_raw_sha256"],
            "source_revision": revision,
            "source_tree_sha256": ancestry["source_tree_sha256"],
            "authorization_sha256": authorization_sha,
            "training_contract_sha256": ancestry["training_contract_sha256"],
            "record": record,
        }
        if (
            not isinstance(integrity, dict)
            or set(integrity) != {*integrity_expected, "self_sha256"}
            or any(
                integrity.get(name) != value
                for name, value in integrity_expected.items()
            )
            or integrity.get("self_sha256")
            != zero_field_identity(integrity, "self_sha256")
        ):
            raise ArtifactIntegrityError(f"seed {seed} integrity record differs")
        panel[seed] = QualifiedCheckpoint(
            seed=seed,
            checkpoint_sha256=str(record["checkpoint_sha256"]),
            replay_path=replay_path,
            replay_sha256=replay_hash,
            replay_rows=replay_rows,
            files=paths,
            file_hashes=hashes,
            component_hashes=checked_components,
        )
    return CheckpointPanel(manifest_path, manifest_sha256, panel)


def verify_checkpoint_manifest_file(
    config: SOCISConfig,
    source_run_directory: str | Path,
    manifest_path: str | Path,
    *,
    expected_authorization_sha256: str | None = None,
) -> CheckpointPanel:
    """Verify an explicit candidate manifest before no-clobber publication."""
    supplied_root = Path(source_run_directory).expanduser()
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ArtifactIntegrityError(
            f"checkpoint source is not a real directory: {supplied_root}"
        )
    root = supplied_root.resolve()
    supplied_manifest = Path(manifest_path)
    if not supplied_manifest.is_absolute():
        supplied_manifest = root / supplied_manifest
    if supplied_manifest.parent.resolve() != root:
        raise ArtifactIntegrityError("checkpoint manifest path escapes source root")
    return _verify_manifest_path(
        config,
        root,
        supplied_manifest,
        expected_authorization_sha256=expected_authorization_sha256,
    )


def verify_checkpoint_manifest(
    config: SOCISConfig,
    source_run_directory: str | Path,
    *,
    expected_authorization_sha256: str | None = None,
) -> CheckpointPanel:
    """Open only the dedicated manifest and its explicit expected file paths."""
    supplied_root = Path(source_run_directory).expanduser()
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ArtifactIntegrityError(
            f"checkpoint source is not a real directory: {supplied_root}"
        )
    root = supplied_root.resolve()
    filename = str(
        config.section("checkpoint_contract")["checkpoint_manifest_filename"]
    )
    manifest_path = root / filename
    return _verify_manifest_path(
        config,
        root,
        manifest_path,
        expected_authorization_sha256=expected_authorization_sha256,
    )


def verify_frozen_checkpoint_manifest(
    config: SOCISConfig,
    source_run_directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_authorization_sha256: str,
) -> CheckpointPanel:
    """Revalidate an immutable Gate 1 panel from a later Phase D source tree."""
    supplied_root = Path(source_run_directory).expanduser()
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ArtifactIntegrityError(
            f"checkpoint source is not a real directory: {supplied_root}"
        )
    root = supplied_root.resolve()
    filename = str(
        config.section("checkpoint_contract")["checkpoint_manifest_filename"]
    )
    return _verify_manifest_path(
        config,
        root,
        root / filename,
        expected_authorization_sha256=expected_authorization_sha256,
        frozen_manifest_sha256=expected_manifest_sha256,
    )


def verify_loaded_checkpoint_components(
    config: SOCISConfig, panel: CheckpointPanel, *, authorized: bool = False
) -> dict[str, dict[str, str]]:
    """Load only verified model paths and recheck all qualified components."""
    if not (
        authorized or bool(config.section("execution")["gpu_execution_authorized"])
    ):
        raise RuntimeError("loaded SOCIS checkpoint verification is not GPU-authorized")
    device = str(config.section("execution")["device_when_authorized"])
    verified: dict[str, dict[str, str]] = {}
    for seed in DEVELOPMENT_SEEDS:
        checkpoint = panel.checkpoints[seed]
        model = SAC.load(checkpoint.files["model"], device=device)
        try:
            model.set_logger(configure(folder=None, format_strings=[]))
            observed = model_component_hashes(model)
            if observed != checkpoint.component_hashes:
                raise ArtifactIntegrityError(
                    f"loaded component hashes differ for checkpoint {seed}"
                )
            verified[str(seed)] = observed
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return verified
