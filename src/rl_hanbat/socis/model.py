"""Frozen SOCIS outcome network and residual-only standardization."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from rl_hanbat.socis.config import SOCISConfig
from rl_hanbat.socis.hashing import canonical_sha256, ndarray_sha256


@dataclass(frozen=True)
class TransitionBatch:
    """Detector-relevant replay arrays with immutable row identities."""

    x: np.ndarray
    continuous_y: np.ndarray
    binary_y: np.ndarray
    row_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, indices: np.ndarray) -> TransitionBatch:
        return TransitionBatch(
            x=self.x[indices],
            continuous_y=self.continuous_y[indices],
            binary_y=self.binary_y[indices],
            row_ids=tuple(self.row_ids[int(index)] for index in indices),
        )


@dataclass(frozen=True)
class RobustLocationScale:
    """Median and floored 1.4826-MAD transform."""

    location: np.ndarray
    scale: np.ndarray
    minimum_scale: float
    sha256: str

    @classmethod
    def fit(cls, values: np.ndarray, minimum_scale: float) -> RobustLocationScale:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("robust standardization requires finite 2D rows")
        location = np.median(array, axis=0)
        mad = np.median(np.abs(array - location), axis=0)
        scale = np.maximum(1.4826 * mad, minimum_scale)
        digest = ndarray_sha256([location, scale])
        return cls(location, scale, minimum_scale, digest)

    def transform(self, values: np.ndarray, *, clip: float | None = None) -> np.ndarray:
        transformed = (
            np.asarray(values, dtype=np.float64) - self.location
        ) / self.scale
        if clip is not None:
            transformed = np.clip(transformed, -clip, clip)
        return transformed


HOPPER_OUTCOME_DIMENSION = 14
"""Hopper-v5's outcome width, and the default so stored bundles load unchanged.

Hopper's state-action width and its outcome width are *both* 14 --- eleven
observations plus three actions, and eleven observation deltas plus a reward
plus two termination flags. A single constant served for both until a second
environment made them differ (Walker2d-v5: 23 and 20). They are separate
quantities and are now named separately.
"""


class OutcomeNetwork(nn.Module):
    """Frozen [256, 256] SiLU network over the outcome width."""

    def __init__(
        self,
        input_dimension: int,
        hidden_layers: list[int],
        output_dimension: int = HOPPER_OUTCOME_DIMENSION,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dimension
        for width in hidden_layers:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, output_dimension))
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def model_sha256(model: nn.Module) -> str:
    """Hash ordered weights without serialization metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _loss(
    prediction: torch.Tensor,
    continuous: torch.Tensor,
    binary: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    split = continuous.shape[1]
    continuous_loss = functional.huber_loss(
        prediction[:, :split], continuous, delta=delta, reduction="mean"
    )
    binary_loss = functional.binary_cross_entropy_with_logits(
        prediction[:, split:], binary, reduction="mean"
    )
    return continuous_loss + binary_loss


@dataclass(frozen=True)
class FittedOutcomeModel:
    """Selected validation-minimum weights and stable-only transforms."""

    network: OutcomeNetwork
    x_scaler: RobustLocationScale
    y_scaler: RobustLocationScale
    residual_scaler: RobustLocationScale
    selected_epoch: int
    model_sha256: str
    preprocessing_sha256: str
    training_record: dict[str, Any]
    device: torch.device
    input_clip: float

    def standardized_x(self, values: np.ndarray) -> np.ndarray:
        return self.x_scaler.transform(values, clip=self.input_clip)

    def raw_residuals(
        self, data: TransitionBatch, batch_size: int = 4096
    ) -> np.ndarray:
        x = self.standardized_x(data.x).astype(np.float32)
        continuous = self.y_scaler.transform(data.continuous_y, clip=self.input_clip)
        outputs: list[np.ndarray] = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                tensor = torch.as_tensor(
                    x[start : start + batch_size], device=self.device
                )
                outputs.append(
                    self.network(tensor).cpu().numpy().astype(np.float64, copy=False)
                )
        prediction = np.concatenate(outputs)
        split = data.continuous_y.shape[1]
        probabilities = 1.0 / (1.0 + np.exp(-prediction[:, split:]))
        return np.concatenate(
            [continuous - prediction[:, :split], data.binary_y - probabilities], axis=1
        )

    def standardized_residuals(self, data: TransitionBatch) -> np.ndarray:
        return self.residual_scaler.transform(self.raw_residuals(data), clip=None)


def _validate_batches(
    training: TransitionBatch,
    validation: TransitionBatch,
    residual_reference: TransitionBatch,
) -> None:
    groups = [
        set(batch.row_ids) for batch in (training, validation, residual_reference)
    ]
    if any(len(batch) == 0 for batch in (training, validation, residual_reference)):
        raise ValueError("outcome model split is empty")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("outcome model replay splits overlap")
    x_width = training.x.shape[1] if training.x.ndim == 2 else 0
    continuous_width = training.continuous_y.shape[1]
    if x_width < 1 or continuous_width < 1:
        raise ValueError("SOCIS X and continuous outcome must be non-degenerate")
    for batch in (training, validation, residual_reference):
        if batch.x.ndim != 2 or batch.x.shape[1] != x_width:
            raise ValueError(f"SOCIS X must have {x_width} dimensions")
        if batch.continuous_y.shape != (len(batch), continuous_width):
            raise ValueError(
                f"SOCIS continuous outcome must have {continuous_width} dimensions")
        if batch.binary_y.shape != (len(batch), 2):
            raise ValueError("SOCIS binary outcome must have two dimensions")
        if not all(
            np.isfinite(values).all()
            for values in (batch.x, batch.continuous_y, batch.binary_y)
        ):
            raise ValueError("outcome model data contain nonfinite values")


def fit_outcome_model(
    config: SOCISConfig,
    training: TransitionBatch,
    validation: TransitionBatch,
    residual_reference: TransitionBatch,
    *,
    seed: int,
    device: str = "cpu",
    gpu_authorized: bool = False,
) -> FittedOutcomeModel:
    """Fit on model_train, select/restore earliest validation minimum, then freeze."""
    _validate_batches(training, validation, residual_reference)
    if device.startswith("cuda") and not (
        gpu_authorized or bool(config.section("execution")["gpu_execution_authorized"])
    ):
        raise RuntimeError("SOCIS GPU execution is not authorized")
    settings = config.section("outcome_model")
    if device.startswith("cuda"):
        expected_workspace = str(settings["cublas_workspace_config"])
        observed_workspace = os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", expected_workspace
        )
        if observed_workspace != expected_workspace:
            raise RuntimeError("SOCIS CUBLAS deterministic workspace differs")
    preprocessing = config.section("preprocessing")
    hidden = [int(value) for value in settings["hidden_layers"]]
    if hidden != [256, 256]:
        raise ValueError("SOCIS outcome network must use hidden layers [256, 256]")
    minimum_scale = float(preprocessing["minimum_scale"])
    input_clip = float(preprocessing["model_input_clip"])
    x_scaler = RobustLocationScale.fit(training.x, minimum_scale)
    y_scaler = RobustLocationScale.fit(training.continuous_y, minimum_scale)
    # Fixed single-thread CPU reductions make bounded verification byte-stable
    # even when another test changed PyTorch's process-wide thread setting.
    if device == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if bool(settings["deterministic_algorithms"]):
        torch.use_deterministic_algorithms(True)
    torch_device = torch.device(device)
    # Continuous outcomes are regressed and binary ones are logits, so the head
    # is as wide as the two together. For Hopper that is 12 + 2 = 14, the value
    # this was written as before a second environment existed.
    outcome_dimension = training.continuous_y.shape[1] + training.binary_y.shape[1]
    network = OutcomeNetwork(
        training.x.shape[1], hidden, outcome_dimension
    ).to(torch_device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=float(settings["learning_rate"])
    )
    train_x = torch.as_tensor(
        x_scaler.transform(training.x, clip=input_clip).astype(np.float32),
        device=torch_device,
    )
    train_y = torch.as_tensor(
        y_scaler.transform(training.continuous_y, clip=input_clip).astype(np.float32),
        device=torch_device,
    )
    train_binary = torch.as_tensor(
        training.binary_y.astype(np.float32), device=torch_device
    )
    validation_x = torch.as_tensor(
        x_scaler.transform(validation.x, clip=input_clip).astype(np.float32),
        device=torch_device,
    )
    validation_y = torch.as_tensor(
        y_scaler.transform(validation.continuous_y, clip=input_clip).astype(np.float32),
        device=torch_device,
    )
    validation_binary = torch.as_tensor(
        validation.binary_y.astype(np.float32), device=torch_device
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch_size = int(settings["batch_size"])
    maximum_epochs = int(settings["maximum_epochs"])
    patience = int(settings["patience"])
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, maximum_epochs + 1):
        network.train()
        order = torch.randperm(len(training), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(
                network(train_x[indices]),
                train_y[indices],
                train_binary[indices],
                delta=float(settings["huber_delta"]),
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        network.eval()
        with torch.no_grad():
            validation_loss = float(
                _loss(
                    network(validation_x),
                    validation_y,
                    validation_binary,
                    delta=float(settings["huber_delta"]),
                ).item()
            )
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / len(training),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in network.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("outcome model did not produce selectable weights")
    network.load_state_dict(best_state)
    fitted_without_residual_scaler = FittedOutcomeModel(
        network=network,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        residual_scaler=RobustLocationScale.fit(
            np.zeros((1, outcome_dimension), dtype=np.float64), minimum_scale
        ),
        selected_epoch=best_epoch,
        model_sha256=model_sha256(network),
        preprocessing_sha256="",
        training_record={},
        device=torch_device,
        input_clip=input_clip,
    )
    raw_reference_residuals = fitted_without_residual_scaler.raw_residuals(
        residual_reference
    )
    residual_scaler = RobustLocationScale.fit(raw_reference_residuals, minimum_scale)
    preprocessing_sha = canonical_sha256(
        {
            "x": x_scaler.sha256,
            "y": y_scaler.sha256,
            "residual": residual_scaler.sha256,
            "input_clip": input_clip,
        }
    )
    record: dict[str, Any] = {
        "seed": seed,
        "model_train_rows": len(training),
        "model_validation_rows": len(validation),
        "residual_reference_rows": len(residual_reference),
        "model_train_identity": canonical_sha256(list(training.row_ids)),
        "model_validation_identity": canonical_sha256(list(validation.row_ids)),
        "residual_reference_identity": canonical_sha256(
            list(residual_reference.row_ids)
        ),
        "selected_epoch": best_epoch,
        "selected_validation_loss": best_loss,
        "selection_tie_break": "earliest_epoch",
        "refit_after_selection": False,
        "history": history,
    }
    return FittedOutcomeModel(
        network=network,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        residual_scaler=residual_scaler,
        selected_epoch=best_epoch,
        model_sha256=model_sha256(network),
        preprocessing_sha256=preprocessing_sha,
        training_record=record,
        device=torch_device,
        input_clip=input_clip,
    )
