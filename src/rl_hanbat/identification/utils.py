"""Deterministic seeds, hashes, manifests, and runtime metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_NAMES = (
    "rl-hanbat",
    "stable-baselines3",
    "gymnasium",
    "mujoco",
    "torch",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "PyYAML",
    "joblib",
    "tabulate",
)


def deterministic_seed(base_seed: int, branch_code: int) -> int:
    """Map a scientific seed and branch code to a stable uint32 seed."""
    sequence = np.random.SeedSequence([base_seed, branch_code])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without silently changing algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | Path) -> str:
    """Hash one file using streaming reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_sha256(parameters: Iterable[torch.nn.Parameter]) -> str:
    """Hash parameter names externally and tensor bytes here."""
    digest = hashlib.sha256()
    for parameter in parameters:
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_component_hashes(model: Any) -> dict[str, str]:
    """Hash the SAC actor, online critics, and target critics separately."""
    return {
        "actor": parameter_sha256(model.actor.parameters()),
        "critics": parameter_sha256(model.critic.parameters()),
        "target_critics": parameter_sha256(model.critic_target.parameters()),
    }


def save_rng_state(path: str | Path) -> None:
    """Persist all RNG states needed to restart a branch."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    torch.save(state, Path(path))


def restore_rng_state(path: str | Path) -> None:
    """Restore a state written by :func:`save_rng_state`."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Write JSON through an adjacent temporary file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def source_revision(repo_root: str | Path) -> str | None:
    """Return the current Git revision when the workspace is versioned."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def package_versions() -> dict[str, str | None]:
    """Return exact installed versions for declared runtime packages."""
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_metadata() -> dict[str, Any]:
    """Return CUDA/GPU facts without requiring nvidia-smi."""
    available = torch.cuda.is_available()
    metadata: dict[str, Any] = {
        "cuda_available": available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
    }
    if available:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "device_index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return metadata


def runtime_metadata() -> dict[str, Any]:
    """Collect environment metadata used by the run manifest."""
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "gpu": gpu_metadata(),
    }


def make_run_id(config_hash: str, seeds: list[int]) -> str:
    """Create a readable unique directory name without affecting protocol identity."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    seed_text = "-".join(str(seed) for seed in seeds)
    return f"{stamp}_{config_hash[:10]}_s{seed_text}"


def relative_to_or_absolute(path: Path, root: Path) -> str:
    """Prefer a portable path relative to the workspace."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def configure_torch_threads() -> None:
    """Avoid accidental thread oversubscription while preserving user overrides."""
    requested = os.environ.get("RL_HANBAT_TORCH_THREADS")
    if requested is not None:
        torch.set_num_threads(max(1, int(requested)))
