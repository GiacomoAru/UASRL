"""ATOM-CBF perception, calibration, and robust safety filter for LIMO.

The implementation follows the adaptive margin and relaxed SOCP in ATOM-CBF.
The perception interface is adapted to the local ray scan used by this project:
an ensemble estimates the closest observed obstacle as ``[distance, bearing]``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - exercised only in an incomplete env
    cp = None


CHECKPOINT_VERSION = 1


class ATOMPerceptionNetwork(nn.Module):
    """Predict normalized closest-obstacle distance and bearing from one scan."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        conv_channels: Sequence[int] = (16, 32),
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        if not conv_channels or any(size < 1 for size in conv_channels):
            raise ValueError("conv_channels must contain positive values")

        convolution: list[nn.Module] = []
        previous_channels = 1
        for channels in conv_channels:
            convolution.extend(
                (
                    nn.Conv1d(previous_channels, int(channels), kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            )
            previous_channels = int(channels)
        self.convolution = nn.Sequential(*convolution)

        head: list[nn.Module] = []
        previous_size = previous_channels * input_dim
        for size in hidden_sizes:
            head.extend((nn.Linear(previous_size, int(size)), nn.ReLU()))
            previous_size = int(size)
        head.append(nn.Linear(previous_size, 2))
        self.head = nn.Sequential(*head)
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.conv_channels = tuple(int(size) for size in conv_channels)

    def forward(self, ray_scan: torch.Tensor) -> torch.Tensor:
        features = self.convolution(ray_scan.unsqueeze(1)).flatten(start_dim=1)
        raw = self.head(features)
        distance = torch.sigmoid(raw[..., :1])
        bearing = torch.tanh(raw[..., 1:2])
        return torch.cat((distance, bearing), dim=-1)


def ray_scan_target(
    ray_scan: np.ndarray,
    ray_angles: np.ndarray,
    ray_length: float,
) -> np.ndarray:
    """Return the closest observed surface point as distance and bearing."""

    scan = np.asarray(ray_scan, dtype=np.float64).reshape(-1)
    angles = np.asarray(ray_angles, dtype=np.float64).reshape(-1)
    if scan.shape != angles.shape:
        raise ValueError("ray_scan and ray_angles must have the same shape")
    if ray_length <= 0:
        raise ValueError("ray_length must be positive")
    if not np.all(np.isfinite(scan)) or not np.all(np.isfinite(angles)):
        raise ValueError("ray_scan and ray_angles must be finite")

    distances = np.clip(scan, 0.0, 1.0) * float(ray_length)
    closest_index = int(np.argmin(distances))
    if distances[closest_index] >= ray_length * (1.0 - 1e-6):
        return np.array([ray_length, 0.0], dtype=np.float64)
    return np.array(
        [distances[closest_index], angles[closest_index]], dtype=np.float64
    )


def normalize_obstacle_targets(
    targets: np.ndarray, ray_length: float, max_abs_bearing: float
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape[-1] != 2:
        raise ValueError("targets must have a final dimension of size 2")
    if ray_length <= 0 or max_abs_bearing <= 0:
        raise ValueError("normalization scales must be positive")
    normalized = targets.copy()
    normalized[..., 0] = np.clip(normalized[..., 0] / ray_length, 0.0, 1.0)
    normalized[..., 1] = np.clip(
        normalized[..., 1] / max_abs_bearing, -1.0, 1.0
    )
    return normalized


def denormalize_obstacle_predictions(
    predictions: np.ndarray, ray_length: float, max_abs_bearing: float
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape[-1] != 2:
        raise ValueError("predictions must have a final dimension of size 2")
    physical = predictions.copy()
    physical[..., 0] *= float(ray_length)
    physical[..., 1] *= float(max_abs_bearing)
    return physical


def deep_ensemble_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """Return the scalar ensemble variance used by ATOM-CBF, per sample."""

    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 3 or predictions.shape[-1] != 2:
        raise ValueError("predictions must have shape [ensemble, samples, 2]")
    mean_prediction = predictions.mean(axis=0)
    mean_squared_norm = np.square(predictions).sum(axis=-1).mean(axis=0)
    squared_mean_norm = np.square(mean_prediction).sum(axis=-1)
    return np.maximum(mean_squared_norm - squared_mean_norm, 0.0)


def wrapped_bearing_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(predicted - target), np.cos(predicted - target))


def calibrate_atom_margin(
    predictions: np.ndarray,
    targets: np.ndarray,
    gamma_multiplier: float = 1.0,
    uncertainty_floor: float = 1e-8,
) -> dict[str, Any]:
    """Calibrate the ATOM-CBF component-wise base error ratio (phi_cal).

    Matches Yun & Azizan, ATOM-CBF, Sec. 4.1 exactly: the calibration set is
    first filtered to `Dfiltered = {(y,x) in Dcal : |Unc(y)-mean| <= gamma}`
    (Eq. 6, `gamma = gamma_multiplier * std`, paper's own default is
    `gamma_multiplier=1.0` i.e. one standard deviation), then phi_cal is the
    per-component worst-case (max) ratio of error to uncertainty over that
    filtered set (Eq. 7). The paper's only defence against a degenerate
    near-zero-uncertainty calibration point inflating that max is the gamma
    filter above -- there is no additional percentile cap or robust floor in
    the paper, so this doesn't add one either; `uncertainty_floor` is purely
    a divide-by-zero guard, not a tuning knob.
    """

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.ndim != 3 or predictions.shape[-1] != 2:
        raise ValueError("predictions must have shape [ensemble, samples, 2]")
    if targets.shape != predictions.shape[1:]:
        raise ValueError("targets must have shape [samples, 2]")
    if predictions.shape[1] < 1:
        raise ValueError("at least one calibration sample is required")
    if gamma_multiplier < 0 or uncertainty_floor <= 0:
        raise ValueError("gamma_multiplier must be nonnegative and floor positive")

    uncertainty = deep_ensemble_uncertainty(predictions)
    uncertainty_mean = float(uncertainty.mean())
    uncertainty_std = float(uncertainty.std(ddof=0))
    gamma = float(gamma_multiplier * uncertainty_std)
    retained = np.abs(uncertainty - uncertainty_mean) <= gamma + 1e-15
    if not np.any(retained):
        retained = np.ones_like(uncertainty, dtype=bool)

    ensemble_mean = predictions.mean(axis=0)
    errors = np.abs(ensemble_mean - targets)
    errors[:, 1] = np.abs(
        wrapped_bearing_error(ensemble_mean[:, 1], targets[:, 1])
    )
    denominator = np.maximum(uncertainty[retained], uncertainty_floor)
    phi_cal = np.max(errors[retained] / denominator[:, None], axis=0)

    return {
        "phi_cal": phi_cal.astype(np.float64),
        "uncertainty_mean": uncertainty_mean,
        "uncertainty_std": uncertainty_std,
        "gamma": gamma,
        "gamma_multiplier": float(gamma_multiplier),
        "uncertainty_floor": float(uncertainty_floor),
        "sample_count": int(uncertainty.size),
        "retained_count": int(retained.sum()),
        "retained_fraction": float(retained.mean()),
        "uncertainty_min": float(uncertainty.min()),
        "uncertainty_max": float(uncertainty.max()),
    }


def cone_barrier_terms(
    distance: float,
    bearing: float,
    obstacle_radius: float,
    nominal_omega: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Return h and Lg h for the cone CBF and the paper's unicycle model."""

    distance = float(distance)
    bearing = float(bearing)
    obstacle_radius = float(obstacle_radius)
    if obstacle_radius <= 0:
        raise ValueError("obstacle_radius must be positive")
    if distance <= obstacle_radius:
        raise ValueError("cone barrier requires distance greater than radius")

    root = math.sqrt(distance * distance - obstacle_radius * obstacle_radius)
    h_distance = obstacle_radius / (distance * root)
    if abs(bearing) > 1e-9:
        bearing_sign = math.copysign(1.0, bearing)
    elif abs(nominal_omega) > 1e-9:
        bearing_sign = math.copysign(1.0, nominal_omega)
    else:
        bearing_sign = 1.0

    h = abs(bearing) - math.asin(obstacle_radius / distance)
    lg_v = (
        -h_distance * math.cos(bearing)
        - bearing_sign * math.sin(bearing) / distance
    )
    lg_omega = bearing_sign
    return float(h), np.array([lg_v, lg_omega], dtype=np.float64)


def estimate_cone_lipschitz_constants(
    obstacle_radius: float,
    min_distance: float,
    max_distance: float,
    max_abs_bearing: float,
    cbf_gain: float,
    distance_samples: int = 320,
    bearing_samples: int = 321,
) -> dict[str, float]:
    """Estimate the Lipschitz constants on a gridded safe local state set."""

    if not obstacle_radius < min_distance < max_distance:
        raise ValueError("expected obstacle_radius < min_distance < max_distance")
    if max_abs_bearing <= 0 or cbf_gain <= 0:
        raise ValueError("bearing range and cbf_gain must be positive")
    if distance_samples < 2 or bearing_samples < 3:
        raise ValueError("the Lipschitz grid is too small")

    distances = np.linspace(min_distance, max_distance, distance_samples)
    bearings = np.linspace(-max_abs_bearing, max_abs_bearing, bearing_samples)
    d_grid, a_grid = np.meshgrid(distances, bearings, indexing="ij")
    safe = np.abs(a_grid) >= np.arcsin(obstacle_radius / d_grid)
    if not np.any(safe):
        raise ValueError("the selected domain contains no cone-CBF safe states")

    sign = np.where(a_grid >= 0.0, 1.0, -1.0)
    root_sq = d_grid * d_grid - obstacle_radius * obstacle_radius
    h_distance = obstacle_radius / (d_grid * np.sqrt(root_sq))
    h_distance_derivative = h_distance * (
        -1.0 / d_grid - d_grid / root_sq
    )
    lg_v_d = (
        -h_distance_derivative * np.cos(a_grid)
        + sign * np.sin(a_grid) / np.square(d_grid)
    )
    lg_v_a = (
        h_distance * np.sin(a_grid)
        - sign * np.cos(a_grid) / d_grid
    )

    h_gradient_norm = np.sqrt(np.square(h_distance) + 1.0)
    lg_gradient_norm = np.sqrt(np.square(lg_v_d) + np.square(lg_v_a))
    l_h = float(np.max(h_gradient_norm[safe]))
    l_lgh = float(np.max(lg_gradient_norm[safe]))
    return {
        "L_Lfh": 0.0,
        "L_h": l_h,
        "L_kappah": float(cbf_gain * l_h),
        "L_Lgh": l_lgh,
        "domain_min_distance": float(min_distance),
        "domain_max_distance": float(max_distance),
        "domain_max_abs_bearing": float(max_abs_bearing),
        "distance_samples": int(distance_samples),
        "bearing_samples": int(bearing_samples),
    }


@dataclass(frozen=True)
class ATOMFilterResult:
    action: np.ndarray
    info: dict[str, Any]


class ATOMCBFController:
    """Adaptive robust cone-CBF filter for normalized LIMO actions."""

    def __init__(
        self,
        models: Sequence[ATOMPerceptionNetwork],
        ray_length: float,
        ray_angles: Sequence[float],
        phi_cal: Sequence[float],
        d_safe: float,
        d_safe_multiplier: float,
        cbf_gain: float,
        lipschitz: dict[str, float],
        max_movement_speed: float,
        max_turn_speed_degrees: float,
        slack_penalty: float = 100.0,
        solver: str = "CLARABEL",
        device: str | torch.device = "cpu",
    ):
        if not models:
            raise ValueError("at least one perception model is required")
        if ray_length <= 0 or d_safe <= 0 or d_safe_multiplier <= 1:
            raise ValueError("ray and safety distances must be positive")
        if cbf_gain <= 0 or slack_penalty <= 0:
            raise ValueError("cbf_gain and slack_penalty must be positive")
        if max_movement_speed <= 0 or max_turn_speed_degrees <= 0:
            raise ValueError("actuator limits must be positive")

        self.device = torch.device(device)
        self.models = list(models)
        for model in self.models:
            model.to(self.device)
            model.eval()
        self.ray_length = float(ray_length)
        self.ray_angles = np.asarray(ray_angles, dtype=np.float64)
        self.max_abs_bearing = float(np.max(np.abs(self.ray_angles)))
        self.phi_cal = np.asarray(phi_cal, dtype=np.float64).reshape(2)
        self.d_safe = float(d_safe)
        self.d_safe_multiplier = float(d_safe_multiplier)
        self.cbf_gain = float(cbf_gain)
        self.lipschitz = {key: float(value) for key, value in lipschitz.items()}
        for required in ("L_Lfh", "L_kappah", "L_Lgh"):
            if required not in self.lipschitz or self.lipschitz[required] < 0:
                raise ValueError(f"missing or invalid Lipschitz constant: {required}")
        self.max_movement_speed = float(max_movement_speed)
        self.max_turn_speed = math.radians(float(max_turn_speed_degrees))
        self.slack_penalty = float(slack_penalty)
        self.solver = str(solver)
        self._build_socp()

    def _build_socp(self) -> None:
        """Build the SOCP once with cp.Parameter placeholders.

        Rebuilding a fresh cp.Problem from scratch on every filter() call
        (the original implementation) re-parses and re-compiles the whole
        symbolic problem each time: ~34ms/call measured, almost entirely
        parsing overhead for a 2-variable QP that should solve in ~1ms.
        At 50Hz physics with several agents that alone exceeds the
        real-time budget. Building the problem once and only updating
        Parameter values per call is the standard cvxpy pattern for
        repeated real-time solves and does not change the solved problem.
        """
        if cp is None:
            self._problem = None
            return

        # cp.installed_solvers() re-scans available solver packages on every
        # call (~16ms measured) -- it doesn't change during a process's
        # lifetime, so cache it once instead of calling it per filter().
        self._installed_solvers = set(cp.installed_solvers())

        self._control = cp.Variable(2)
        self._slack = cp.Variable(nonneg=True)
        self._nominal_param = cp.Parameter(2)
        self._barrier_param = cp.Parameter()
        self._lg_h_param = cp.Parameter(2)
        self._epsilon_param = cp.Parameter(nonneg=True)

        fixed_lipschitz = self.lipschitz["L_Lfh"] + self.lipschitz["L_kappah"]
        robust_margin = self._epsilon_param * (
            fixed_lipschitz + self.lipschitz["L_Lgh"] * cp.norm(self._control, 2)
        )
        constraints = [
            self._lg_h_param @ self._control - robust_margin
            >= -self.cbf_gain * self._barrier_param - self._slack,
            self._control[0] >= 0.0,
            self._control[0] <= self.max_movement_speed,
            self._control[1] >= -self.max_turn_speed,
            self._control[1] <= self.max_turn_speed,
        ]
        # Plain ||u - u_nominal||^2, matching the paper's relaxed filter
        # exactly (Yun & Azizan, ATOM-CBF, Eq. 15): no actuator-normalized
        # D^-1 weighting here. (That normalization belongs to the project's
        # own rebuttal-specified CBF, testing_utils.py -- it isn't part of
        # ATOM-CBF and was wrongly ported over here.)
        objective = cp.Minimize(
            0.5 * cp.sum_squares(self._control - self._nominal_param)
            + self.slack_penalty * cp.square(self._slack)
        )
        self._problem = cp.Problem(objective, constraints)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        max_movement_speed: float,
        max_turn_speed_degrees: float,
        device: str | torch.device = "cpu",
        expected_ray_length: float | None = None,
        expected_ray_angles: Sequence[float] | None = None,
        expected_d_safe: float | None = None,
        expected_d_safe_multiplier: float | None = None,
        expected_cbf_gain: float | None = None,
        solver: str | None = None,
    ) -> "ATOMCBFController":
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported ATOM-CBF checkpoint version")
        config = checkpoint["config"]

        checks = (
            ("ray_length", expected_ray_length),
            ("d_safe", expected_d_safe),
            ("d_safe_multiplier", expected_d_safe_multiplier),
            ("cbf_gain", expected_cbf_gain),
        )
        for key, expected in checks:
            if expected is not None and not math.isclose(
                float(config[key]), float(expected), rel_tol=1e-7, abs_tol=1e-9
            ):
                raise ValueError(
                    f"checkpoint {key}={config[key]} differs from runtime {expected}"
                )
        if expected_ray_angles is not None and not np.allclose(
            np.asarray(config["ray_angles"], dtype=np.float64),
            np.asarray(expected_ray_angles, dtype=np.float64),
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError("checkpoint ray angles differ from the runtime sensor")

        models: list[ATOMPerceptionNetwork] = []
        for state_dict in checkpoint["model_state_dicts"]:
            model = ATOMPerceptionNetwork(
                config["input_dim"],
                config["hidden_sizes"],
                config.get("conv_channels", (16, 32)),
            )
            model.load_state_dict(state_dict)
            models.append(model)

        return cls(
            models=models,
            ray_length=config["ray_length"],
            ray_angles=config["ray_angles"],
            phi_cal=checkpoint["calibration"]["phi_cal"],
            d_safe=config["d_safe"],
            d_safe_multiplier=config["d_safe_multiplier"],
            cbf_gain=config["cbf_gain"],
            lipschitz=checkpoint["lipschitz"],
            max_movement_speed=max_movement_speed,
            max_turn_speed_degrees=max_turn_speed_degrees,
            slack_penalty=config.get("slack_penalty", 100.0),
            solver=solver or config.get("solver", "CLARABEL"),
            device=device,
        )

    def predict_obstacle(
        self, ray_scan: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        scan = np.asarray(ray_scan, dtype=np.float32).reshape(-1)
        if scan.shape != self.ray_angles.shape:
            raise ValueError(
                f"expected {self.ray_angles.size} rays, received {scan.size}"
            )
        if not np.all(np.isfinite(scan)):
            raise ValueError("ray scan must be finite")
        scan = np.clip(scan, 0.0, 1.0)
        ray_tensor = torch.as_tensor(scan, device=self.device).unsqueeze(0)
        with torch.no_grad():
            normalized = torch.stack(
                [model(ray_tensor)[0] for model in self.models], dim=0
            ).cpu().numpy()
        predictions = denormalize_obstacle_predictions(
            normalized, self.ray_length, self.max_abs_bearing
        )
        mean_prediction = predictions.mean(axis=0)
        uncertainty = float(
            deep_ensemble_uncertainty(predictions[:, None, :])[0]
        )
        return mean_prediction, predictions, uncertainty

    def filter(self, ray_scan: np.ndarray, nominal_action: np.ndarray) -> ATOMFilterResult:
        started_at = time.perf_counter()
        action = np.asarray(nominal_action, dtype=np.float64).reshape(-1)
        if action.size != 2 or not np.all(np.isfinite(action)):
            raise ValueError("nominal_action must contain two finite values")

        nominal_physical = np.array(
            [
                np.clip(action[0], 0.0, 1.0) * self.max_movement_speed,
                np.clip(action[1], -1.0, 1.0) * self.max_turn_speed,
            ],
            dtype=np.float64,
        )
        obstacle, ensemble_predictions, uncertainty = self.predict_obstacle(ray_scan)
        epsilon_adapt = float(np.linalg.norm(self.phi_cal * uncertainty, ord=2))
        base_info: dict[str, Any] = {
            "estimated_obstacle": obstacle.copy(),
            "ensemble_predictions": ensemble_predictions.copy(),
            "uncertainty": uncertainty,
            "epsilon_adapt": epsilon_adapt,
            "phi_cal": self.phi_cal.copy(),
            "solver": self.solver,
            "slack": 0.0,
        }

        if obstacle[0] > self.d_safe * self.d_safe_multiplier:
            normalized = self._normalize_action(nominal_physical)
            base_info.update(
                {
                    "status": "inactive_distance",
                    "barrier": None,
                    "activated": bool(np.linalg.norm(normalized - action) > 1e-6),
                    "solve_time": time.perf_counter() - started_at,
                }
            )
            return ATOMFilterResult(normalized, base_info)

        if obstacle[0] <= self.d_safe + 1e-6:
            fallback = np.array([0.0, nominal_physical[1]], dtype=np.float64)
            normalized = self._normalize_action(fallback)
            base_info.update(
                {
                    "status": "emergency_inside_margin",
                    "barrier": None,
                    "activated": bool(np.linalg.norm(normalized - action) > 1e-6),
                    "solve_time": time.perf_counter() - started_at,
                }
            )
            return ATOMFilterResult(normalized, base_info)

        h, lg_h = cone_barrier_terms(
            obstacle[0], obstacle[1], self.d_safe, nominal_physical[1]
        )
        physical, status, slack, objective = self._solve_socp(
            nominal_physical, h, lg_h, epsilon_adapt
        )
        normalized = self._normalize_action(physical)
        base_info.update(
            {
                "status": status,
                "barrier": h,
                "lg_h": lg_h.copy(),
                "slack": slack,
                "objective": objective,
                "activated": bool(np.linalg.norm(normalized - action) > 1e-6),
                "solve_time": time.perf_counter() - started_at,
            }
        )
        return ATOMFilterResult(normalized, base_info)

    def _solve_socp(
        self,
        nominal_physical: np.ndarray,
        barrier: float,
        lg_h: np.ndarray,
        epsilon_adapt: float,
    ) -> tuple[np.ndarray, str, float, float | None]:
        if cp is None or self._problem is None:
            raise RuntimeError(
                "ATOM-CBF requires cvxpy. Install the project requirements first."
            )

        self._nominal_param.value = np.asarray(nominal_physical, dtype=np.float64)
        self._barrier_param.value = float(barrier)
        self._lg_h_param.value = np.asarray(lg_h, dtype=np.float64)
        self._epsilon_param.value = max(0.0, float(epsilon_adapt))

        attempted: list[str] = []
        for solver in (self.solver, "CLARABEL", "SCS"):
            if solver in attempted or solver not in self._installed_solvers:
                continue
            attempted.append(solver)
            try:
                self._problem.solve(solver=solver, warm_start=True, verbose=False)
            except cp.error.SolverError:
                continue
            if self._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                value = np.asarray(self._control.value, dtype=np.float64).reshape(2)
                value[0] = np.clip(value[0], 0.0, self.max_movement_speed)
                value[1] = np.clip(
                    value[1], -self.max_turn_speed, self.max_turn_speed
                )
                return (
                    value,
                    str(self._problem.status),
                    float(max(0.0, self._slack.value)),
                    float(self._problem.value),
                )

        fallback = np.array([0.0, nominal_physical[1]], dtype=np.float64)
        return fallback, "solver_failure", 0.0, None

    def _normalize_action(self, physical_action: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip(
                    physical_action[0] / self.max_movement_speed, 0.0, 1.0
                ),
                np.clip(physical_action[1] / self.max_turn_speed, -1.0, 1.0),
            ],
            dtype=np.float64,
        )


def predict_ensemble_physical(
    models: Iterable[ATOMPerceptionNetwork],
    scans: np.ndarray,
    ray_length: float,
    max_abs_bearing: float,
    device: str | torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Run an ensemble and return predictions shaped [M, N, 2]."""

    scans = np.asarray(scans, dtype=np.float32)
    if scans.ndim != 2:
        raise ValueError("scans must have shape [samples, rays]")
    device = torch.device(device)
    all_predictions: list[np.ndarray] = []
    for model in models:
        model.to(device)
        model.eval()
        model_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, scans.shape[0], batch_size):
                batch = torch.as_tensor(scans[start : start + batch_size], device=device)
                model_predictions.append(model(batch).cpu().numpy())
        normalized = np.concatenate(model_predictions, axis=0)
        all_predictions.append(
            denormalize_obstacle_predictions(
                normalized, ray_length, max_abs_bearing
            )
        )
    return np.stack(all_predictions, axis=0)
