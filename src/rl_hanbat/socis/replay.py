"""Deterministic, disjoint, exhaustive stable replay splitting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rl_hanbat.socis.checkpoints import CheckpointPanel
from rl_hanbat.socis.config import SOCISConfig
from rl_hanbat.socis.hashing import canonical_json_bytes, file_sha256, source_revision
from rl_hanbat.socis.model import TransitionBatch
from rl_hanbat.socis.persistence import (
    ArtifactIntegrityError,
    atomic_write_json,
    atomic_write_parquet,
    require_regular_file,
    validate_completed_parquet,
    validate_run_directory_entries,
)
from rl_hanbat.socis.schema import replay_split_schema

SPLIT_NAMES = ("model_train", "model_validation", "residual_reference")
_INTERRUPTED_REPLAY = re.compile(
    r"^\.seed_(?:3[0-9])\.(?:parquet|integrity\.json)\.interrupted_[0-9]+$"
)


@dataclass(frozen=True)
class ReplaySplit:
    """Persistable assignments plus exact row indices for each split."""

    assignments: pa.Table
    indices: dict[str, np.ndarray]


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_jsonable(child) for child in value]
    return value


def replay_row_sha256(table: pa.Table, index: int) -> str:
    """Hash one immutable replay row through canonical Arrow scalar values."""
    row = {name: _jsonable(table[name][index].as_py()) for name in table.column_names}
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()


def split_replay_table(
    config: SOCISConfig,
    table: pa.Table,
    *,
    checkpoint_seed: int,
    checkpoint_sha256: str,
    replay_sha256: str,
) -> ReplaySplit:
    """Hash-order rows and assign floor-60/floor-20/remainder partitions."""
    if checkpoint_seed not in config.development_seeds:
        raise ValueError("replay checkpoint seed is outside development 30--39")
    count = len(table)
    if count < 3:
        raise ValueError("stable replay needs at least three rows")
    prefix = str(config.section("replay_split")["hash_prefix"])
    row_ids = [f"{checkpoint_seed}:{index}" for index in range(count)]
    row_hashes = [replay_row_sha256(table, index) for index in range(count)]
    ranks = [
        hashlib.sha256(
            (
                prefix + "|" + str(checkpoint_seed) + "|" + row_id + "|" + row_hash
            ).encode("utf-8")
        ).hexdigest()
        for row_id, row_hash in zip(row_ids, row_hashes, strict=True)
    ]
    order = np.argsort(np.asarray(ranks, dtype=object), kind="stable")
    train_stop = int(np.floor(0.60 * count))
    validation_stop = train_stop + int(np.floor(0.20 * count))
    split_by_index = np.empty(count, dtype=object)
    split_by_index[order[:train_stop]] = "model_train"
    split_by_index[order[train_stop:validation_stop]] = "model_validation"
    split_by_index[order[validation_stop:]] = "residual_reference"
    indices = {
        name: np.flatnonzero(split_by_index == name).astype(np.int64)
        for name in SPLIT_NAMES
    }
    combined = np.concatenate(list(indices.values()))
    if len(np.unique(combined)) != count or set(combined.tolist()) != set(range(count)):
        raise RuntimeError("replay split is not disjoint and exhaustive")
    assignments = pa.Table.from_arrays(
        [
            pa.array([checkpoint_seed] * count, type=pa.int16()),
            pa.array(row_ids, type=pa.string()),
            pa.array(row_hashes, type=pa.string()),
            pa.array(split_by_index.tolist(), type=pa.string()),
            pa.array(ranks, type=pa.string()),
            pa.array([checkpoint_sha256] * count, type=pa.string()),
            pa.array([replay_sha256] * count, type=pa.string()),
        ],
        schema=replay_split_schema(),
    )
    return ReplaySplit(assignments=assignments, indices=indices)


def verify_replay_split(
    split: ReplaySplit,
    expected_rows: int,
    *,
    checkpoint_seed: int | None = None,
    checkpoint_sha256: str | None = None,
    replay_sha256: str | None = None,
) -> None:
    """Recheck schema, split names, disjointness, and exhaustion on resume."""
    if split.assignments.schema.remove_metadata() != replay_split_schema():
        raise ValueError("replay split assignment schema differs")
    if len(split.assignments) != expected_rows:
        raise ValueError("replay split assignment row count differs")
    if set(split.indices) != set(SPLIT_NAMES):
        raise ValueError("replay split names differ")
    combined = np.concatenate([split.indices[name] for name in SPLIT_NAMES])
    if len(combined) != expected_rows or len(np.unique(combined)) != expected_rows:
        raise ValueError("replay split is not disjoint")
    if set(combined.tolist()) != set(range(expected_rows)):
        raise ValueError("replay split is not exhaustive")
    expected_counts = {
        "model_train": int(np.floor(0.60 * expected_rows)),
        "model_validation": int(np.floor(0.20 * expected_rows)),
    }
    expected_counts["residual_reference"] = expected_rows - sum(
        expected_counts.values()
    )
    if {name: len(split.indices[name]) for name in SPLIT_NAMES} != expected_counts:
        raise ValueError("replay split fractions differ")
    rows = split.assignments
    if checkpoint_seed is not None:
        if set(rows["checkpoint_seed"].to_pylist()) != {checkpoint_seed}:
            raise ValueError("replay split checkpoint seed differs")
        if rows["replay_row_id"].to_pylist() != [
            f"{checkpoint_seed}:{index}" for index in range(expected_rows)
        ]:
            raise ValueError("replay split row identities differ")
    for column in ("row_sha256", "split_rank_sha256"):
        values = rows[column].to_pylist()
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ):
            raise ValueError(f"replay split {column} values differ")
    if len(set(rows["split_rank_sha256"].to_pylist())) != expected_rows:
        raise ValueError("replay split ranks are not unique")
    if checkpoint_sha256 is not None and set(rows["checkpoint_sha256"].to_pylist()) != {
        checkpoint_sha256
    }:
        raise ValueError("replay split checkpoint identity differs")
    if replay_sha256 is not None and set(rows["replay_sha256"].to_pylist()) != {
        replay_sha256
    }:
        raise ValueError("replay split source identity differs")


def _fixed_values(column: pa.ChunkedArray) -> np.ndarray:
    combined = column.combine_chunks()
    if pa.types.is_fixed_size_list(combined.type):
        return combined.values.to_numpy(zero_copy_only=False).reshape(
            len(combined), combined.type.list_size
        )
    return combined.to_numpy(zero_copy_only=False)


def extract_transition_batch(
    table: pa.Table,
    assignments: pa.Table,
    *,
    split_name: str,
    checkpoint_seed: int,
) -> TransitionBatch:
    """Extract one frozen split without admitting changed or monitoring rows."""
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown replay split: {split_name}")
    required = {
        "observation",
        "action",
        "next_observation",
        "reward",
        "terminated",
        "truncated",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"stable replay columns are missing: {sorted(missing)}")
    if len(table) != len(assignments):
        raise ValueError("replay rows and assignments differ")
    if "condition" in table.column_names:
        conditions = set(str(value) for value in table["condition"].to_pylist())
        if not conditions <= {"stable_frozen", "stable"}:
            raise ValueError("changed rows are forbidden in checkpoint replay")
    if "checkpoint_seed" in table.column_names:
        seeds = set(int(value) for value in table["checkpoint_seed"].to_pylist())
        if seeds != {checkpoint_seed}:
            raise ValueError("replay record checkpoint seed differs")
    split_values = np.asarray(assignments["split"].to_pylist(), dtype=object)
    indices = np.flatnonzero(split_values == split_name).astype(np.int64)
    observation = _fixed_values(table["observation"])[indices].astype(np.float64)
    action = _fixed_values(table["action"])[indices].astype(np.float64)
    next_observation = _fixed_values(table["next_observation"])[indices].astype(
        np.float64
    )
    reward = table["reward"].to_numpy(zero_copy_only=False)[indices].astype(np.float64)
    terminated = table["terminated"].to_numpy(zero_copy_only=False)[indices]
    truncated = table["truncated"].to_numpy(zero_copy_only=False)[indices]
    row_ids_all = assignments["replay_row_id"].to_pylist()
    return TransitionBatch(
        x=np.concatenate([observation, action], axis=1),
        continuous_y=np.concatenate(
            [next_observation - observation, reward[:, None]], axis=1
        ),
        binary_y=np.stack([terminated, truncated], axis=1).astype(np.float64),
        row_ids=tuple(str(row_ids_all[int(index)]) for index in indices),
    )


def _split_from_assignments(assignments: pa.Table) -> ReplaySplit:
    values = np.asarray(assignments["split"].to_pylist(), dtype=object)
    return ReplaySplit(
        assignments=assignments,
        indices={
            name: np.flatnonzero(values == name).astype(np.int64)
            for name in SPLIT_NAMES
        },
    )


def lock_panel_replay_splits(
    config: SOCISConfig, panel: CheckpointPanel, run_directory: Path
) -> dict[str, Any]:
    """Persist or verify all ten stable-replay assignments without other access."""
    root = run_directory / "replay_splits"
    expected_names = {
        name
        for seed in config.development_seeds
        for name in (f"seed_{seed}.parquet", f"seed_{seed}.integrity.json")
    }
    if root.exists() and root.is_dir() and not root.is_symlink():
        expected_names |= {
            entry.name
            for entry in root.iterdir()
            if _INTERRUPTED_REPLAY.fullmatch(entry.name)
        }
    validate_run_directory_entries(root, expected_names)
    root.mkdir(parents=True, exist_ok=True)
    revision = source_revision(config.workspace)
    if revision is None:
        raise ArtifactIntegrityError("SOCIS source revision is unavailable")
    records: dict[str, Any] = {}
    for seed in config.development_seeds:
        checkpoint = panel.checkpoints[seed]
        assignments_path = root / f"seed_{seed}.parquet"
        integrity_path = root / f"seed_{seed}.integrity.json"
        metadata = {
            "schema_version": "1",
            "config_sha256": config.sha256,
            "checkpoint_seed": str(seed),
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "replay_sha256": checkpoint.replay_sha256,
            "source_manifest_sha256": panel.manifest_sha256,
            "source_revision": revision,
        }
        assignments_present = assignments_path.exists() or assignments_path.is_symlink()
        integrity_present = integrity_path.exists() or integrity_path.is_symlink()
        if assignments_present != integrity_present:
            existing = assignments_path if assignments_present else integrity_path
            require_regular_file(existing)
            preserved = root / f".{existing.name}.interrupted_{time.time_ns()}"
            os.replace(existing, preserved)
            assignments_present = False
            integrity_present = False
        if assignments_present or integrity_present:
            require_regular_file(assignments_path)
            require_regular_file(integrity_path)
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            if not isinstance(integrity, dict):
                raise ArtifactIntegrityError(
                    f"replay split integrity root differs for seed {seed}"
                )
            expected_integrity_keys = {
                "schema_version",
                "source_revision",
                "config_sha256",
                "checkpoint_seed",
                "checkpoint_sha256",
                "replay_sha256",
                "source_manifest_sha256",
                "rows",
                "split_counts",
                "disjoint",
                "exhaustive",
                "assignments_sha256",
            }
            if set(integrity) != expected_integrity_keys:
                raise ArtifactIntegrityError(
                    f"replay split integrity schema differs for seed {seed}"
                )
            expected_split_counts = {
                "model_train": int(np.floor(0.60 * checkpoint.replay_rows)),
                "model_validation": int(np.floor(0.20 * checkpoint.replay_rows)),
            }
            expected_split_counts["residual_reference"] = checkpoint.replay_rows - sum(
                expected_split_counts.values()
            )
            exact = {
                "schema_version": 1,
                "source_revision": revision,
                "config_sha256": config.sha256,
                "checkpoint_seed": seed,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "replay_sha256": checkpoint.replay_sha256,
                "source_manifest_sha256": panel.manifest_sha256,
                "rows": checkpoint.replay_rows,
                "split_counts": expected_split_counts,
                "disjoint": True,
                "exhaustive": True,
            }
            if any(integrity.get(key) != value for key, value in exact.items()):
                raise ArtifactIntegrityError(
                    f"replay split integrity differs for seed {seed}"
                )
            expected_hash = integrity.get("assignments_sha256")
            if not isinstance(expected_hash, str):
                raise ArtifactIntegrityError(
                    f"replay split hash is missing for seed {seed}"
                )
            assignments = validate_completed_parquet(
                assignments_path,
                expected_schema=replay_split_schema(),
                expected_rows=checkpoint.replay_rows,
                expected_sha256=expected_hash,
                expected_metadata=metadata,
            )
            verify_replay_split(
                _split_from_assignments(assignments),
                checkpoint.replay_rows,
                checkpoint_seed=seed,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                replay_sha256=checkpoint.replay_sha256,
            )
            if file_sha256(checkpoint.replay_path) != checkpoint.replay_sha256:
                raise ArtifactIntegrityError(
                    f"stable replay hash differs for seed {seed}"
                )
            stable_replay = pq.read_table(checkpoint.replay_path)
            expected_split = split_replay_table(
                config,
                stable_replay,
                checkpoint_seed=seed,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                replay_sha256=checkpoint.replay_sha256,
            )
            if not assignments.replace_schema_metadata(None).equals(
                expected_split.assignments
            ):
                raise ArtifactIntegrityError(
                    f"replay split assignments differ for seed {seed}"
                )
            records[str(seed)] = integrity
            continue

        stable_replay = pq.read_table(checkpoint.replay_path)
        if len(stable_replay) != checkpoint.replay_rows:
            raise ArtifactIntegrityError(
                f"stable replay row count differs for seed {seed}"
            )
        if file_sha256(checkpoint.replay_path) != checkpoint.replay_sha256:
            raise ArtifactIntegrityError(f"stable replay hash differs for seed {seed}")
        split = split_replay_table(
            config,
            stable_replay,
            checkpoint_seed=seed,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            replay_sha256=checkpoint.replay_sha256,
        )
        verify_replay_split(
            split,
            checkpoint.replay_rows,
            checkpoint_seed=seed,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            replay_sha256=checkpoint.replay_sha256,
        )
        encoded_metadata = {
            key.encode(): value.encode() for key, value in metadata.items()
        }
        assignments = split.assignments.replace_schema_metadata(encoded_metadata)
        digest = atomic_write_parquet(assignments_path, assignments)
        integrity = {
            "schema_version": 1,
            "source_revision": revision,
            "config_sha256": config.sha256,
            "checkpoint_seed": seed,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "replay_sha256": checkpoint.replay_sha256,
            "source_manifest_sha256": panel.manifest_sha256,
            "rows": checkpoint.replay_rows,
            "split_counts": {name: len(split.indices[name]) for name in SPLIT_NAMES},
            "disjoint": True,
            "exhaustive": True,
            "assignments_sha256": digest,
        }
        atomic_write_json(integrity_path, integrity)
        records[str(seed)] = integrity
    return {
        "scientific_passed": True,
        "source_manifest_sha256": panel.manifest_sha256,
        "checkpoint_seeds": list(config.development_seeds),
        "records": records,
    }
