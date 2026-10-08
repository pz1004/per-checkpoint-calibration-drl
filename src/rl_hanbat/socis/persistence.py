"""Fail-closed, fsynced, atomic persistence and resumability checks."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rl_hanbat.socis.hashing import file_sha256


class ArtifactIntegrityError(RuntimeError):
    """Raised when a SOCIS artifact cannot be safely accepted."""


def require_regular_file(path: Path) -> None:
    """Reject missing, symlinked, and non-regular artifact paths."""
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactIntegrityError(f"missing artifact: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ArtifactIntegrityError(f"artifact is not a regular file: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write bytes through a same-directory temporary file and fsync both."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Write auditable JSON atomically without accepting NaN values."""
    payload = (
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_parquet(path: str | Path, table: pa.Table) -> str:
    """Write one Parquet shard atomically and return its raw SHA-256."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = file_sha256(temporary)
        os.replace(temporary, output)
        _fsync_directory(output.parent)
        return digest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> str:
    """Write a non-pickle compressed NumPy bundle atomically and hash it."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
            handle.flush()
            os.fsync(handle.fileno())
        digest = file_sha256(temporary)
        os.replace(temporary, output)
        _fsync_directory(output.parent)
        return digest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_completed_parquet(
    path: Path,
    *,
    expected_schema: pa.Schema,
    expected_rows: int,
    expected_sha256: str,
    expected_metadata: dict[str, str],
) -> pa.Table:
    """Accept a completed shard only after all locked resume checks pass."""
    require_regular_file(path)
    if file_sha256(path) != expected_sha256:
        raise ArtifactIntegrityError(f"artifact hash mismatch: {path}")
    table = pq.read_table(path)
    if table.schema.remove_metadata() != expected_schema.remove_metadata():
        raise ArtifactIntegrityError(f"artifact schema mismatch: {path}")
    if len(table) != expected_rows:
        raise ArtifactIntegrityError(f"artifact row-count mismatch: {path}")
    metadata = table.schema.metadata or {}
    for key, value in expected_metadata.items():
        if metadata.get(key.encode()) != value.encode():
            raise ArtifactIntegrityError(
                f"artifact metadata mismatch for {key}: {path}"
            )
    return table


def validate_run_directory_entries(path: Path, expected: set[str]) -> None:
    """Reject unknown or symlinked top-level resume content."""
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise ArtifactIntegrityError(f"run path is not a real directory: {path}")
    unknown: list[str] = []
    for entry in path.iterdir():
        if entry.is_symlink() or entry.name not in expected:
            unknown.append(entry.name)
    if unknown:
        raise ArtifactIntegrityError(
            "unknown or symlinked run content: " + ", ".join(sorted(unknown))
        )
