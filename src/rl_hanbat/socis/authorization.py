"""Read-only Phase C authorization verification before checkpoint access."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_hanbat.socis.config import DEVELOPMENT_SEEDS, SOCISConfig
from rl_hanbat.socis.hashing import (
    file_sha256,
    source_revision,
    source_tree_sha256,
    zero_field_identity,
)
from rl_hanbat.socis.persistence import ArtifactIntegrityError, require_regular_file


class PhaseCAuthorizationError(ArtifactIntegrityError):
    """Raised before Phase C output or checkpoint paths may be inspected."""


AUTHORIZATION_FIELDS = {
    "schema_version",
    "purpose",
    "decision",
    "protocol_name",
    "protocol_version",
    "protocol_amendment",
    "operational_revision",
    "config_sha256",
    "config_raw_sha256",
    "method_proposal_sha256",
    "amendment_sha256",
    "phase_c_operational_lock_sha256",
    "source_revision",
    "source_tree_sha256",
    "workspace",
    "config_path",
    "output_root",
    "checkpoint_seeds",
    "device",
    "canonical_command",
    "issued_at_utc",
    "expires_at_utc",
    "human_approver",
    "rationale",
    "authorization_scope",
    "development_collection_authorized",
    "confirmation_authorized",
    "self_sha256",
}


@dataclass(frozen=True)
class PhaseCAuthorization:
    """Verified Phase C authority bound to one output root and checkout."""

    path: Path
    self_sha256: str
    raw_sha256: str
    output_root: Path
    device: str
    issued_at: datetime
    expires_at: datetime
    human_approver: str
    rationale: str
    data: dict[str, Any]


def authorization_identity(data: dict[str, Any]) -> str:
    """Return the canonical zero-field identity of an authorization record."""
    return zero_field_identity(data, "self_sha256")


def _utc_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise PhaseCAuthorizationError(f"Phase C {name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PhaseCAuthorizationError(
            f"Phase C {name} must be an ISO timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PhaseCAuthorizationError(f"Phase C {name} must include a timezone")
    return parsed.astimezone(UTC)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseCAuthorizationError(f"Phase C {name} must be nonempty")
    return value


def _expected_output_root(config: SOCISConfig, supplied: str | Path) -> Path:
    candidate = Path(supplied).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise PhaseCAuthorizationError("Phase C output root must be absolute")
    settings = config.section("phase_c_authorization")
    pattern = str(settings["output_root_name_pattern"])
    if re.fullmatch(pattern, candidate.name) is None:
        raise PhaseCAuthorizationError("Phase C output root name differs")
    parent = config.workspace / str(settings["output_root_parent"])
    expected = parent / candidate.name
    if candidate != expected:
        raise PhaseCAuthorizationError(
            "Phase C output root must be a direct child of the locked run root"
        )
    return candidate


def verify_phase_c_authorization(
    path: str | Path,
    config: SOCISConfig,
    output_root: str | Path,
    *,
    now: datetime | None = None,
) -> PhaseCAuthorization:
    """Verify all Phase C authority before touching the bound output root."""
    authorization_path = Path(path).expanduser()
    require_regular_file(authorization_path)
    try:
        loaded = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseCAuthorizationError(
            "Phase C authorization is not valid JSON"
        ) from error
    if not isinstance(loaded, dict) or set(loaded) != AUTHORIZATION_FIELDS:
        raise PhaseCAuthorizationError("Phase C authorization schema differs")
    settings = config.section("phase_c_authorization")
    protocol = config.section("protocol")
    expected_root = _expected_output_root(config, output_root)
    revision = source_revision(config.workspace)
    if revision is None:
        raise PhaseCAuthorizationError("Phase C source revision is unavailable")
    exact: dict[str, Any] = {
        "schema_version": settings["schema_version"],
        "purpose": settings["purpose"],
        "decision": settings["decision"],
        "protocol_name": protocol["name"],
        "protocol_version": protocol["version"],
        "protocol_amendment": protocol["amendment"],
        "operational_revision": protocol["operational_revision"],
        "config_sha256": config.sha256,
        "config_raw_sha256": config.raw_sha256,
        "method_proposal_sha256": protocol["method_proposal_sha256"],
        "amendment_sha256": protocol["amendment_sha256"],
        "phase_c_operational_lock_sha256": protocol["phase_c_operational_lock_sha256"],
        "source_revision": revision,
        "source_tree_sha256": source_tree_sha256(config.workspace),
        "workspace": str(config.workspace.resolve()),
        "config_path": str(config.path),
        "output_root": str(expected_root),
        "checkpoint_seeds": list(DEVELOPMENT_SEEDS),
        "device": settings["device"],
        "canonical_command": settings["canonical_command"],
        "authorization_scope": settings["authorization_scope"],
        "development_collection_authorized": False,
        "confirmation_authorized": False,
    }
    for name, expected in exact.items():
        if loaded.get(name) != expected:
            raise PhaseCAuthorizationError(f"Phase C authorization {name} differs")
    recorded = loaded.get("self_sha256")
    if not isinstance(recorded, str) or recorded != authorization_identity(loaded):
        raise PhaseCAuthorizationError("Phase C authorization self-hash mismatch")
    issued = _utc_datetime(loaded.get("issued_at_utc"), "issued_at_utc")
    expires = _utc_datetime(loaded.get("expires_at_utc"), "expires_at_utc")
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    if not issued <= current < expires:
        raise PhaseCAuthorizationError("Phase C authorization is not currently valid")
    approver = _nonempty(loaded.get("human_approver"), "human approver")
    rationale = _nonempty(loaded.get("rationale"), "rationale")
    return PhaseCAuthorization(
        path=authorization_path.resolve(),
        self_sha256=recorded,
        raw_sha256=file_sha256(authorization_path),
        output_root=expected_root,
        device=str(settings["device"]),
        issued_at=issued,
        expires_at=expires,
        human_approver=approver,
        rationale=rationale,
        data=loaded,
    )
