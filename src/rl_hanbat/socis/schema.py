"""Exact Arrow schemas for SOCIS transitions, decisions, and replay splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from rl_hanbat.socis.config import CONDITIONS, SOCISConfig
from rl_hanbat.socis.hashing import canonical_sha256

BLOCK_STATES = {"WARMUP", "MONITORING", "ABSTAIN_SUPPORT", "ALARM", "POST_ALARM"}


def checkpoint_replay_schema(config: SOCISConfig) -> pa.Schema:
    """Return the exact qualified stable-training replay export schema."""
    contract = config.section("checkpoint_training")["stable_replay"]
    observation_dimension = int(contract["observation_dimension"])
    action_dimension = int(contract["action_dimension"])
    return pa.schema(
        [
            pa.field("checkpoint_seed", pa.int16(), nullable=False),
            pa.field("replay_row_index", pa.int64(), nullable=False),
            pa.field("condition", pa.string(), nullable=False),
            pa.field(
                "observation",
                pa.list_(pa.float64(), observation_dimension),
                nullable=False,
            ),
            pa.field(
                "action", pa.list_(pa.float64(), action_dimension), nullable=False
            ),
            pa.field(
                "next_observation",
                pa.list_(pa.float64(), observation_dimension),
                nullable=False,
            ),
            pa.field("reward", pa.float64(), nullable=False),
            pa.field("terminated", pa.bool_(), nullable=False),
            pa.field("truncated", pa.bool_(), nullable=False),
        ]
    )


def arrow_schema_identity(schema: pa.Schema) -> str:
    """Hash Arrow names, types, and nullability without mutable metadata."""
    return canonical_sha256(
        [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema.remove_metadata()
        ]
    )


def transition_schema(config: SOCISConfig) -> pa.Schema:
    """Return the immutable raw transition record contract."""
    contract = config.section("schemas")["transition_record"]
    observation_dimension = int(contract["observation_dimension"])
    action_dimension = int(contract["action_dimension"])
    return pa.schema(
        [
            pa.field("stream_id", pa.string(), nullable=False),
            pa.field("condition", pa.string(), nullable=False),
            pa.field("policy_mode", pa.string(), nullable=False),
            pa.field("checkpoint_seed", pa.int16(), nullable=False),
            pa.field("stream_seed", pa.int64(), nullable=False),
            pa.field("replicate", pa.int8(), nullable=False),
            pa.field("window_index", pa.int16(), nullable=False),
            pa.field("transition_index", pa.int16(), nullable=False),
            pa.field("global_transition_index", pa.int32(), nullable=False),
            pa.field(
                "observation",
                pa.list_(pa.float64(), observation_dimension),
                nullable=False,
            ),
            pa.field(
                "next_observation",
                pa.list_(pa.float64(), observation_dimension),
                nullable=False,
            ),
            pa.field(
                "action", pa.list_(pa.float64(), action_dimension), nullable=False
            ),
            pa.field("reward", pa.float64(), nullable=False),
            pa.field("terminated", pa.bool_(), nullable=False),
            pa.field("truncated", pa.bool_(), nullable=False),
            pa.field("policy_version", pa.int64(), nullable=False),
            pa.field("actor_update_count", pa.int64(), nullable=False),
            pa.field("critic_update_count", pa.int64(), nullable=False),
            pa.field("gravity_multiplier", pa.float64(), nullable=False),
            pa.field("checkpoint_sha256", pa.string(), nullable=False),
            pa.field("config_sha256", pa.string(), nullable=False),
            pa.field("seed_code_version", pa.string(), nullable=False),
        ]
    )


def block_decision_schema() -> pa.Schema:
    """Return the exact one-row-per-window detector decision schema."""

    def nullable_float(name: str) -> pa.Field:
        return pa.field(name, pa.float64(), nullable=True)

    return pa.schema(
        [
            pa.field("stream_id", pa.string(), nullable=False),
            pa.field("checkpoint_seed", pa.int16(), nullable=False),
            pa.field("replicate", pa.int8(), nullable=False),
            pa.field("condition", pa.string(), nullable=False),
            pa.field("policy_mode", pa.string(), nullable=False),
            pa.field("window_index", pa.int16(), nullable=False),
            pa.field("monitoring_window", pa.int16(), nullable=True),
            pa.field("supported_mass", pa.float64(), nullable=False),
            pa.field("current_ess", pa.float64(), nullable=False),
            pa.field("reference_ess", pa.float64(), nullable=False),
            pa.field("support_boundary", pa.float64(), nullable=False),
            nullable_float("mmd_u_rff"),
            nullable_float("null_mean"),
            nullable_float("null_std"),
            nullable_float("z_score"),
            pa.field("glr_value", pa.float64(), nullable=False),
            pa.field("alarm_threshold", pa.float64(), nullable=False),
            pa.field("state", pa.string(), nullable=False),
            pa.field("reason_code", pa.string(), nullable=False),
            pa.field("alarm", pa.bool_(), nullable=False),
            pa.field("preprocessing_sha256", pa.string(), nullable=False),
            pa.field("model_sha256", pa.string(), nullable=False),
            pa.field("reference_sha256", pa.string(), nullable=False),
            pa.field("rff_sha256", pa.string(), nullable=False),
            pa.field("standardizer_sha256", pa.string(), nullable=False),
        ]
    )


def replay_split_schema() -> pa.Schema:
    """Return the persisted replay-assignment schema."""
    return pa.schema(
        [
            pa.field("checkpoint_seed", pa.int16(), nullable=False),
            pa.field("replay_row_id", pa.string(), nullable=False),
            pa.field("row_sha256", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("split_rank_sha256", pa.string(), nullable=False),
            pa.field("checkpoint_sha256", pa.string(), nullable=False),
            pa.field("replay_sha256", pa.string(), nullable=False),
        ]
    )


def _fixed_list(values: np.ndarray, dimension: int) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.float64).reshape(-1), pa.float64())
    return pa.FixedSizeListArray.from_arrays(flat, dimension)


def transition_table(
    config: SOCISConfig,
    arrays: dict[str, np.ndarray],
    *,
    metadata: dict[str, str],
) -> pa.Table:
    """Construct an exact-schema table and reject invalid regimes or shapes."""
    schema = transition_schema(config)
    count = len(arrays["stream_id"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("transition columns have inconsistent lengths")
    unknown = set(str(value) for value in arrays["condition"]) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown SOCIS conditions: {sorted(unknown)}")
    contract = config.section("schemas")["transition_record"]
    observation_dimension = int(contract["observation_dimension"])
    action_dimension = int(contract["action_dimension"])
    columns: list[pa.Array] = []
    for field in schema:
        if field.name in {"observation", "next_observation"}:
            column = _fixed_list(arrays[field.name], observation_dimension)
        elif field.name == "action":
            column = _fixed_list(arrays[field.name], action_dimension)
        else:
            column = pa.array(arrays[field.name], type=field.type)
        columns.append(column)
    encoded = {key.encode(): value.encode() for key, value in metadata.items()}
    return pa.Table.from_arrays(columns, schema=schema.with_metadata(encoded))


@dataclass(frozen=True)
class BlockDecision:
    """Typed representation of one SOCIS block state."""

    stream_id: str
    checkpoint_seed: int
    replicate: int
    condition: str
    policy_mode: str
    window_index: int
    monitoring_window: int | None
    supported_mass: float
    current_ess: float
    reference_ess: float
    support_boundary: float
    mmd_u_rff: float | None
    null_mean: float | None
    null_std: float | None
    z_score: float | None
    glr_value: float
    alarm_threshold: float
    state: str
    reason_code: str
    alarm: bool
    preprocessing_sha256: str
    model_sha256: str
    reference_sha256: str
    rff_sha256: str
    standardizer_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def block_decision_table(
    decisions: list[BlockDecision] | tuple[BlockDecision, ...],
    *,
    metadata: dict[str, str],
) -> pa.Table:
    """Build one exact decision shard and enforce state-dependent nullability."""
    if not decisions:
        raise ValueError("decision shard must contain at least one row")
    expected_windows = list(range(len(decisions)))
    if [row.window_index for row in decisions] != expected_windows:
        raise ValueError("decision windows must be contiguous and zero based")
    alarm_seen = False
    for row in decisions:
        if row.condition not in CONDITIONS:
            raise ValueError(f"unknown decision condition: {row.condition}")
        if row.policy_mode not in {"frozen", "updating"}:
            raise ValueError(f"unknown decision policy mode: {row.policy_mode}")
        if row.state not in BLOCK_STATES or not row.reason_code:
            raise ValueError("decision state or reason code differs")
        scores = (row.mmd_u_rff, row.null_mean, row.null_std, row.z_score)
        if row.state == "WARMUP":
            if row.monitoring_window is not None or any(
                value is not None for value in scores
            ):
                raise ValueError("warm-up decision contains monitoring scores")
        else:
            if row.monitoring_window is None or row.monitoring_window < 1:
                raise ValueError("monitoring decision lacks a 1-based window")
            if row.state == "ABSTAIN_SUPPORT":
                if any(value is not None for value in scores):
                    raise ValueError("support abstention contains a score")
            elif any(value is None for value in scores):
                raise ValueError("supported monitoring decision lacks a score")
        if row.state == "ALARM":
            if alarm_seen or not row.alarm:
                raise ValueError("decision shard must contain one first alarm")
            alarm_seen = True
        elif row.alarm:
            raise ValueError("only the first ALARM row may set alarm=true")
        if row.state == "POST_ALARM" and not alarm_seen:
            raise ValueError("POST_ALARM cannot precede ALARM")
        for name in (
            "preprocessing_sha256",
            "model_sha256",
            "reference_sha256",
            "rff_sha256",
            "standardizer_sha256",
        ):
            value = str(getattr(row, name))
            if len(value) != 64:
                raise ValueError(f"decision {name} is not SHA-256")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"decision {name} is not hexadecimal") from error
    encoded = {key.encode(): value.encode() for key, value in metadata.items()}
    schema = block_decision_schema().with_metadata(encoded)
    return pa.Table.from_pylist([row.as_dict() for row in decisions], schema=schema)
