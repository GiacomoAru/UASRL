"""Local MPPI navigation baseline for LIMO.

Second baseline from PROSSIME_MODIFICHE_ICRA.md: a sampling-based (MPPI,
Williams et al.) local planner that *replaces* the nominal policy entirely.
It gets exactly what the policy gets and nothing else:

- the local ray scan (same field of view / ray_length as the policy sensor);
- kinematic state (forward/lateral/angular velocity);
- the goal expressed relative to the robot (distance, angle).

No global map, no full environment state, no unobservable obstacles: the
ray hits are treated as fixed local obstacle points for the rollout horizon
(the environment is assumed locally static over that short horizon, same
simplification the project's own CBF/ATOM-CBF baselines make).

Sign-convention note (verified against the Unity source, not assumed):
Unity's `Vector3.SignedAngle(forward, to, Vector3.up)` (used for
`angle_to_goal`) and `Rigidbody.angularVelocity.y` (used for the executed
`omega` command) are both positive-clockwise-from-above ("positive = turn
right"). The ray sensor's own angle convention
(`testing_utils.generate_angles_rad`) is positive-left instead. Internally
this module rolls out in a standard right-handed local frame (x=forward,
y=left, heading positive=CCW=left) to reuse the same obstacle-point math as
the CBF/ATOM code, and flips sign only at the two boundaries where a
Unity-convention quantity crosses in (goal angle) or out (the omega command
applied to the kinematic model). The public in/out interface (bearing in
radians for obstacles, `angle_to_goal` in radians, action normalized
[-1, 1]) matches Unity's native convention throughout, so callers never see
the flip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MPPIResult:
    action: np.ndarray
    info: dict[str, Any]


class MPPILocalPlanner:
    """Sampling-based local planner: local ray scan -> (v, omega)."""

    def __init__(
        self,
        ray_length: float,
        ray_angles: np.ndarray,
        max_movement_speed: float,
        max_turn_speed_degrees: float,
        horizon: int = 15,
        num_samples: int = 512,
        dt: float = 0.1,
        temperature: float = 0.2,
        noise_std: tuple[float, float] = (0.4, 0.6),
        obstacle_radius: float = 0.25,
        w_goal: float = 4.0,
        w_obstacle: float = 3.0,
        w_smooth: float = 0.05,
        collision_penalty: float = 400.0,
        seed: int = 0,
    ):
        if ray_length <= 0 or max_movement_speed <= 0 or max_turn_speed_degrees <= 0:
            raise ValueError("ray_length and actuator limits must be positive")
        if horizon < 1 or num_samples < 1:
            raise ValueError("horizon and num_samples must be positive")
        if dt <= 0 or temperature <= 0 or obstacle_radius <= 0:
            raise ValueError("dt, temperature and obstacle_radius must be positive")

        self.ray_length = float(ray_length)
        self.ray_angles = np.asarray(ray_angles, dtype=np.float64).reshape(-1)
        self.max_v = float(max_movement_speed)
        self.max_omega = math.radians(float(max_turn_speed_degrees))
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.dt = float(dt)
        self.temperature = float(temperature)
        self.noise_std = (float(noise_std[0]), float(noise_std[1]))
        self.obstacle_radius = float(obstacle_radius)
        self.w_goal = float(w_goal)
        self.w_obstacle = float(w_obstacle)
        self.w_smooth = float(w_smooth)
        self.collision_penalty = float(collision_penalty)
        self.rng = np.random.default_rng(seed)

        # Warm-started mean control sequence (Unity convention: v>=0, omega
        # signed clockwise-positive), shifted forward each call.
        self._mean_u = np.zeros((self.horizon, 2), dtype=np.float64)

    def reset(self) -> None:
        """Call at episode start so the warm-started sequence doesn't leak."""
        self._mean_u[:] = 0.0

    def plan(
        self,
        ray_obs: np.ndarray,
        forward_speed_norm: float,
        angular_speed_norm: float,
        distance_to_goal: float,
        angle_to_goal_rad: float,
    ) -> MPPIResult:
        """One MPPI step. All inputs are exactly what the policy observes.

        `angle_to_goal_rad` and the returned `omega` follow Unity's own
        convention (positive = clockwise/right); ray obstacles follow the
        project's ray-angle convention (positive = left).
        """
        ray_obs = np.asarray(ray_obs, dtype=np.float64).reshape(-1)
        if ray_obs.shape != self.ray_angles.shape:
            raise ValueError(
                f"expected {self.ray_angles.size} rays, received {ray_obs.size}"
            )
        if not np.all(np.isfinite(ray_obs)):
            raise ValueError("ray_obs must be finite")

        # Local obstacle points, ray convention (x=forward, y=left).
        distances = np.clip(ray_obs, 0.0, 1.0) * self.ray_length
        obstacle_xy = np.column_stack(
            (distances * np.cos(self.ray_angles), distances * np.sin(self.ray_angles))
        )
        # Only rays that actually hit something within range are real
        # obstacles; rays at max range carry no local information.
        hit = distances < self.ray_length * (1.0 - 1e-6)
        obstacle_xy = obstacle_xy[hit]

        # Goal point in the same (x=forward, y=left) frame: flip the sign
        # once here, at the boundary, per the module-level convention note.
        goal_bearing_left = -float(angle_to_goal_rad)
        goal_xy = float(distance_to_goal) * np.array(
            [math.cos(goal_bearing_left), math.sin(goal_bearing_left)]
        )

        # Sample K candidate control sequences around the warm-started mean.
        noise = self.rng.normal(
            0.0,
            self.noise_std,
            size=(self.num_samples, self.horizon, 2),
        )
        candidates = self._mean_u[None, :, :] + noise
        candidates[..., 0] = np.clip(candidates[..., 0], 0.0, self.max_v)
        candidates[..., 1] = np.clip(candidates[..., 1], -self.max_omega, self.max_omega)

        costs = self._rollout_cost(candidates, obstacle_xy, goal_xy)

        # MPPI update: softmax-weighted average of the sampled sequences.
        beta = float(costs.min())
        weights = np.exp(-(costs - beta) / self.temperature)
        weights_sum = weights.sum()
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            weights = np.ones_like(weights)
            weights_sum = weights.sum()
        weights /= weights_sum
        self._mean_u = np.tensordot(weights, candidates, axes=(0, 0))
        self._mean_u[:, 0] = np.clip(self._mean_u[:, 0], 0.0, self.max_v)
        self._mean_u[:, 1] = np.clip(self._mean_u[:, 1], -self.max_omega, self.max_omega)

        v, omega = self._mean_u[0]
        action = np.array(
            [np.clip(v / self.max_v, 0.0, 1.0), np.clip(omega / self.max_omega, -1.0, 1.0)],
            dtype=np.float64,
        )

        # Shift the mean sequence forward (receding horizon warm start);
        # repeat the last control to fill the freed slot.
        self._mean_u = np.roll(self._mean_u, -1, axis=0)
        self._mean_u[-1] = self._mean_u[-2]

        return MPPIResult(
            action,
            {
                "cost_min": float(beta),
                "cost_mean": float(costs.mean()),
                "effective_samples": float(1.0 / np.sum(weights**2)),
                "num_obstacle_points": int(obstacle_xy.shape[0]),
            },
        )

    def _rollout_cost(
        self, candidates: np.ndarray, obstacle_xy: np.ndarray, goal_xy: np.ndarray
    ) -> np.ndarray:
        """Vectorized rollout + cost over all samples. candidates: [K,T,2]."""

        k = candidates.shape[0]
        pose = np.zeros((k, 3), dtype=np.float64)  # x, y, heading (CCW+)
        cost = np.zeros(k, dtype=np.float64)

        for t in range(self.horizon):
            v = candidates[:, t, 0]
            omega_unity = candidates[:, t, 1]
            heading = pose[:, 2]

            # Kinematic model, CCW-positive local frame; Unity's omega is
            # clockwise-positive, hence the sign flip on the heading update.
            pose[:, 0] += v * np.cos(heading) * self.dt
            pose[:, 1] += v * np.sin(heading) * self.dt
            pose[:, 2] -= omega_unity * self.dt

            if obstacle_xy.shape[0] > 0:
                delta = pose[:, None, 0:2] - obstacle_xy[None, :, :]
                dist = np.linalg.norm(delta, axis=-1)
                min_dist = dist.min(axis=1)
                clearance = self.obstacle_radius - min_dist
                cost += self.w_obstacle * np.square(np.maximum(clearance, 0.0))
                cost += self.collision_penalty * (min_dist < self.obstacle_radius)

            if t > 0:
                cost += self.w_smooth * np.sum(
                    np.square(candidates[:, t, :] - candidates[:, t - 1, :]), axis=-1
                )

        to_goal = goal_xy[None, :] - pose[:, 0:2]
        cost += self.w_goal * np.linalg.norm(to_goal, axis=-1)
        return cost


def mppi_from_obs(
    ray_obs: np.ndarray,
    ray_original_length: float,
    max_movement_speed: float,
    max_turn_speed: float,
    forward_speed_norm: float,
    angular_speed_norm: float,
    distance_to_goal: float,
    angle_to_goal_deg: float,
    planner: MPPILocalPlanner,
) -> np.ndarray:
    """Thin adapter matching the project's *_from_obs(...) -> normalized action
    convention (see CBF_from_obs / ECBF_from_obs in testing_utils.py)."""

    result = planner.plan(
        ray_obs,
        forward_speed_norm,
        angular_speed_norm,
        distance_to_goal,
        math.radians(angle_to_goal_deg),
    )
    return result.action
