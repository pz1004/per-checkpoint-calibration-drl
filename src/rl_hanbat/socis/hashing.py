"""Canonical identities used by the isolated SOCIS-RL package."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ZERO_SHA256 = "0" * 64


def canonical_json_bytes(data: Any) -> bytes:
    """Serialize JSON-compatible data with the protocol's canonical encoding."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    """Hash a regular file without following a caller-supplied directory scan."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(data: Any) -> str:
    """Hash canonical JSON data."""
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def zero_field_identity(data: dict[str, Any], dotted_field: str) -> str:
    """Hash a deep copy after replacing one dotted field with 64 zeroes."""
    payload = json.loads(json.dumps(data, allow_nan=False))
    current: Any = payload
    parts = dotted_field.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_field)
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(dotted_field)
    current[parts[-1]] = ZERO_SHA256
    return canonical_sha256(payload)


def ndarray_sha256(arrays: Iterable[np.ndarray]) -> str:
    """Hash typed, shaped array bytes in their supplied order."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def source_revision(workspace: Path) -> str | None:
    """Read the local Git revision without changing repository state."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def source_tree_sha256(workspace: Path) -> str:
    """Hash the exact executable and locked operational source inventory."""
    source_root = workspace / "src" / "rl_hanbat"
    candidates = list(source_root.rglob("*.py"))
    candidates.extend(
        [
            workspace / "pyproject.toml",
            workspace / "uv.lock",
            workspace / "experiments" / "socis_rl" / "config.yaml",
            workspace / "experiments" / "socis_rl" / "METHOD_PROPOSAL.md",
            workspace / "experiments" / "socis_rl" / "PROTOCOL_AMENDMENT_1.md",
            workspace / "experiments" / "socis_rl" / "PROTOCOL_AMENDMENT_2.md",
            workspace / "experiments" / "socis_rl" / "PHASE_C_OPERATIONAL_LOCK.md",
            workspace / "experiments" / "socis_rl" / "PHASE_D_OPERATIONAL_LOCK.md",
        ]
    )
    records: list[dict[str, str]] = []
    for path in sorted(set(candidates)):
        try:
            info = path.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"missing source identity path: {path}") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"invalid source identity path: {path}")
        records.append(
            {
                "path": str(path.relative_to(workspace)),
                "sha256": file_sha256(path),
            }
        )
    if not records:
        raise ValueError("source identity inventory is empty")
    return canonical_sha256(records)
