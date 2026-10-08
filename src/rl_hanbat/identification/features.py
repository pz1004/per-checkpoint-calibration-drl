"""Window-level trajectory and policy-update instrumentation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

TRAJECTORY_FEATURES = (
    "mean_reward",
    "mean_min_critic_value",
    "median_signed_td_error",
    "log_iqr_abs_td_error",
    "mean_action_change_norm",
    "action_direction_reversal_rate",
)

UPDATE_FEATURES = (
    "relative_actor_parameter_change",
    "relative_critic_parameter_change",
    "probe_mean_action_displacement",
    "actor_symmetric_kl",
    "actor_entropy_change",
)


@dataclass(frozen=True)
class ParameterSnapshot:
    """Flattened actor and critic parameters at a window boundary."""

    actor: torch.Tensor
    critic: torch.Tensor


@dataclass(frozen=True)
class ActorProbeSnapshot:
    """Pre-squash actor distribution and deterministic actions on fixed states."""

    mean: np.ndarray
    log_std: np.ndarray
    mean_action: np.ndarray
    entropy: float


@dataclass
class WindowAccumulator:
    """Collect transition features for exactly one fixed-size window."""

    rewards: list[float] = field(default_factory=list)
    min_q_values: list[float] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    action_change_norms: list[float] = field(default_factory=list)
    action_reversals: list[float] = field(default_factory=list)
    reset_count: int = 0
    instrumentation_seconds: float = 0.0
    _previous_action: np.ndarray | None = None
    _previous_action_delta: np.ndarray | None = None

    def add(
        self,
        *,
        reward: float,
        min_q_value: float,
        td_error: float,
        action: np.ndarray,
        instrumentation_seconds: float,
    ) -> None:
        """Add one transition and derive action-change features."""
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        self.rewards.append(float(reward))
        self.min_q_values.append(float(min_q_value))
        self.td_errors.append(float(td_error))
        self.instrumentation_seconds += float(instrumentation_seconds)

        if self._previous_action is not None:
            delta = action_array - self._previous_action
            self.action_change_norms.append(float(np.linalg.norm(delta)))
            if self._previous_action_delta is not None:
                dot = float(np.dot(delta, self._previous_action_delta))
                self.action_reversals.append(float(dot < 0.0))
            self._previous_action_delta = delta
        self._previous_action = action_array.copy()

    def note_reset(self) -> None:
        """Log an episode boundary and prevent across-reset action comparisons."""
        self.reset_count += 1
        self._previous_action = None
        self._previous_action_delta = None

    def finalize(self, expected_transitions: int) -> dict[str, float | int]:
        """Return the six locked trajectory features and integrity metadata."""
        if len(self.rewards) != expected_transitions:
            raise RuntimeError(
                f"Window has {len(self.rewards)} transitions; "
                f"expected {expected_transitions}"
            )
        absolute_td = np.abs(np.asarray(self.td_errors, dtype=np.float64))
        iqr = float(np.quantile(absolute_td, 0.75) - np.quantile(absolute_td, 0.25))
        return {
            "mean_reward": float(np.mean(self.rewards)),
            "mean_min_critic_value": float(np.mean(self.min_q_values)),
            "median_signed_td_error": float(np.median(self.td_errors)),
            "log_iqr_abs_td_error": float(np.log(max(iqr, np.finfo(float).tiny))),
            "mean_action_change_norm": _mean_or_zero(self.action_change_norms),
            "action_direction_reversal_rate": _mean_or_zero(self.action_reversals),
            "reset_count": self.reset_count,
            "instrumentation_seconds": self.instrumentation_seconds,
        }


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _flatten_parameters(module: torch.nn.Module) -> torch.Tensor:
    tensors = [
        parameter.detach().cpu().reshape(-1).to(dtype=torch.float64)
        for parameter in module.parameters()
    ]
    return torch.cat(tensors) if tensors else torch.empty(0, dtype=torch.float64)


def parameter_snapshot(model: Any) -> ParameterSnapshot:
    """Capture online actor and critic parameters."""
    return ParameterSnapshot(
        actor=_flatten_parameters(model.actor),
        critic=_flatten_parameters(model.critic),
    )


def _relative_change(before: torch.Tensor, after: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(after - before)
    denominator = torch.linalg.vector_norm(before).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return float((numerator / denominator).item())


def actor_probe_snapshot(model: Any, probe_states: np.ndarray) -> ActorProbeSnapshot:
    """Evaluate the SAC actor distribution without modifying RNG state."""
    observations, _ = model.policy.obs_to_tensor(probe_states)
    with torch.no_grad():
        mean, log_std, _ = model.actor.get_action_dist_params(observations)
        normalized_action = model.actor(observations, deterministic=True)
    mean_array = mean.detach().cpu().numpy().astype(np.float64, copy=False)
    log_std_array = log_std.detach().cpu().numpy().astype(np.float64, copy=False)
    normalized_array = normalized_action.detach().cpu().numpy()
    mean_action = model.policy.unscale_action(normalized_array).astype(
        np.float64, copy=False
    )
    entropy = float(
        np.mean(
            np.sum(
                log_std_array + 0.5 * math.log(2.0 * math.pi * math.e),
                axis=1,
            )
        )
    )
    return ActorProbeSnapshot(
        mean=mean_array,
        log_std=log_std_array,
        mean_action=mean_action,
        entropy=entropy,
    )


def update_features(
    parameter_before: ParameterSnapshot,
    parameter_after: ParameterSnapshot,
    probe_before: ActorProbeSnapshot,
    probe_after: ActorProbeSnapshot,
) -> dict[str, float]:
    """Compute the five locked window-level update features."""
    action_displacement = np.linalg.norm(
        probe_after.mean_action - probe_before.mean_action,
        axis=1,
    )
    symmetric_kl = _symmetric_diagonal_gaussian_kl(
        probe_before.mean,
        probe_before.log_std,
        probe_after.mean,
        probe_after.log_std,
    )
    return {
        "relative_actor_parameter_change": _relative_change(
            parameter_before.actor, parameter_after.actor
        ),
        "relative_critic_parameter_change": _relative_change(
            parameter_before.critic, parameter_after.critic
        ),
        "probe_mean_action_displacement": float(np.mean(action_displacement)),
        "actor_symmetric_kl": float(np.mean(symmetric_kl)),
        "actor_entropy_change": float(probe_after.entropy - probe_before.entropy),
    }


def _diagonal_gaussian_kl(
    mean_a: np.ndarray,
    log_std_a: np.ndarray,
    mean_b: np.ndarray,
    log_std_b: np.ndarray,
) -> np.ndarray:
    variance_a = np.exp(2.0 * log_std_a)
    variance_b = np.exp(2.0 * log_std_b)
    terms = (
        2.0 * (log_std_b - log_std_a)
        + (variance_a + np.square(mean_a - mean_b)) / variance_b
        - 1.0
    )
    return 0.5 * np.sum(terms, axis=1)


def _symmetric_diagonal_gaussian_kl(
    mean_a: np.ndarray,
    log_std_a: np.ndarray,
    mean_b: np.ndarray,
    log_std_b: np.ndarray,
) -> np.ndarray:
    forward = _diagonal_gaussian_kl(mean_a, log_std_a, mean_b, log_std_b)
    reverse = _diagonal_gaussian_kl(mean_b, log_std_b, mean_a, log_std_a)
    return 0.5 * (forward + reverse)


def zero_update_features() -> dict[str, float]:
    """Return the exact frozen-policy update-feature vector."""
    return {name: 0.0 for name in UPDATE_FEATURES}


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def sac_transition_diagnostics(
    model: Any,
    observation: np.ndarray,
    action: np.ndarray,
    next_observation: np.ndarray,
    reward: float,
    terminated: bool,
    measurement_seed: int,
) -> tuple[float, float, float]:
    """Return min-Q, signed TD error, and instrumentation wall time.

    The target critic is evaluated under ``no_grad``. A terminated transition
    receives no bootstrap term; a time-limit truncation is deliberately not an
    argument and therefore keeps the bootstrap term.
    """
    started = time.perf_counter()
    device = model.device
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(
        1, -1
    )
    act = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
    next_obs = torch.as_tensor(
        next_observation, dtype=torch.float32, device=device
    ).reshape(1, -1)
    with torch.no_grad():
        online_q = model.critic(obs, act)
        min_online_q = torch.minimum(online_q[0], online_q[1])
        with torch.random.fork_rng(devices=_fork_rng_devices(device)):
            torch.manual_seed(measurement_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(measurement_seed)
            next_action, next_log_prob = model.actor.action_log_prob(next_obs)
        target_q = model.critic_target(next_obs, next_action)
        min_target_q = torch.minimum(target_q[0], target_q[1])
        entropy_coefficient = _entropy_coefficient(model)
        bootstrap = min_target_q - entropy_coefficient * next_log_prob.reshape(-1, 1)
        target = float(reward) + model.gamma * (1.0 - float(terminated)) * bootstrap
        signed_td_error = target - min_online_q
    elapsed = time.perf_counter() - started
    return (
        float(min_online_q.item()),
        float(signed_td_error.item()),
        elapsed,
    )


def scalar_sac_td_target(
    *,
    reward: float,
    gamma: float,
    terminated: bool,
    truncated: bool,
    minimum_target_q: float,
    entropy_coefficient: float,
    next_log_probability: float,
) -> float:
    """Reference SAC target used to test terminal/truncation semantics.

    ``truncated`` is explicit to make the intended nonterminal treatment
    auditable even though it does not enter the bootstrap mask.
    """
    del truncated
    bootstrap = minimum_target_q - entropy_coefficient * next_log_probability
    return reward + gamma * (1.0 - float(terminated)) * bootstrap


def _entropy_coefficient(model: Any) -> torch.Tensor:
    if model.log_ent_coef is not None:
        return torch.exp(model.log_ent_coef.detach())
    if model.ent_coef_tensor is not None:
        return model.ent_coef_tensor.detach()
    raise RuntimeError("SAC entropy coefficient is unavailable")
