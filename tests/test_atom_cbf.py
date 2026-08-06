import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from atom_cbf import (
    ATOMCBFController,
    ATOMPerceptionNetwork,
    CHECKPOINT_VERSION,
    calibrate_atom_margin,
    cone_barrier_terms,
    deep_ensemble_uncertainty,
    estimate_cone_lipschitz_constants,
)
from testing_utils import generate_angles_rad


class TestATOMCalibration(unittest.TestCase):
    def test_deep_ensemble_uncertainty_matches_definition(self):
        predictions = np.array(
            [
                [[1.0, 0.0], [2.0, 1.0]],
                [[3.0, 0.0], [2.0, -1.0]],
            ]
        )
        uncertainty = deep_ensemble_uncertainty(predictions)
        np.testing.assert_allclose(uncertainty, [1.0, 1.0])

    def test_calibration_uses_component_error_over_uncertainty(self):
        predictions = np.array(
            [
                [[0.4, 0.1], [0.7, -0.3]],
                [[0.6, 0.1], [0.9, -0.3]],
            ]
        )
        targets = np.array([[0.55, 0.2], [0.75, -0.1]])
        calibration = calibrate_atom_margin(
            predictions, targets, gamma_multiplier=1e6
        )
        uncertainty = deep_ensemble_uncertainty(predictions)
        mean = predictions.mean(axis=0)
        errors = np.abs(mean - targets)
        expected_phi = np.max(errors / uncertainty[:, None], axis=0)
        np.testing.assert_allclose(calibration["phi_cal"], expected_phi)
        self.assertEqual(calibration["retained_count"], 2)


class TestConeBarrier(unittest.TestCase):
    def test_lie_derivative_matches_finite_difference(self):
        distance = 0.9
        bearing = 0.45
        radius = 0.25
        control = np.array([0.6, -0.2])
        h, lg_h = cone_barrier_terms(distance, bearing, radius, control[1])

        dt = 1e-7
        distance_dot = -math.cos(bearing) * control[0]
        bearing_dot = -math.sin(bearing) / distance * control[0] + control[1]
        next_h = abs(bearing + dt * bearing_dot) - math.asin(
            radius / (distance + dt * distance_dot)
        )
        numerical_derivative = (next_h - h) / dt
        self.assertAlmostEqual(numerical_derivative, float(lg_h @ control), places=6)

    def test_lipschitz_estimate_is_positive(self):
        constants = estimate_cone_lipschitz_constants(
            obstacle_radius=0.25,
            min_distance=0.30,
            max_distance=0.75,
            max_abs_bearing=math.pi / 2,
            cbf_gain=1.5,
            distance_samples=32,
            bearing_samples=33,
        )
        self.assertEqual(constants["L_Lfh"], 0.0)
        self.assertGreater(constants["L_kappah"], 0.0)
        self.assertGreater(constants["L_Lgh"], 0.0)


class TestATOMController(unittest.TestCase):
    def setUp(self):
        self.angles = np.asarray(generate_angles_rad(10, 90), dtype=np.float64)
        self.models = [ATOMPerceptionNetwork(21, [8]) for _ in range(2)]
        self.lipschitz = {"L_Lfh": 0.0, "L_kappah": 2.0, "L_Lgh": 0.4}
        self.controller = ATOMCBFController(
            models=self.models,
            ray_length=3.0,
            ray_angles=self.angles,
            phi_cal=[0.0, 0.0],
            d_safe=0.25,
            d_safe_multiplier=3.0,
            cbf_gain=1.5,
            lipschitz=self.lipschitz,
            max_movement_speed=1.0,
            max_turn_speed_degrees=92.0,
        )

    def test_socp_respects_bounds_and_relaxed_constraint(self):
        nominal = np.array([0.8, 0.0])
        h, lg_h = cone_barrier_terms(0.5, 0.1, 0.25)
        physical, status, slack, _ = self.controller._solve_socp(
            nominal, h, lg_h, epsilon_adapt=0.08
        )
        self.assertIn(status, ("optimal", "optimal_inaccurate"))
        self.assertGreaterEqual(physical[0], -1e-8)
        self.assertLessEqual(physical[0], 1.0 + 1e-8)
        self.assertLessEqual(abs(physical[1]), math.radians(92.0) + 1e-8)
        robust_margin = 0.08 * (
            self.lipschitz["L_Lfh"]
            + self.lipschitz["L_kappah"]
            + self.lipschitz["L_Lgh"] * np.linalg.norm(physical)
        )
        lhs = float(lg_h @ physical - robust_margin)
        rhs = -1.5 * h - slack
        self.assertGreaterEqual(lhs + 1e-5, rhs)

    def test_adaptive_margin_changes_the_optimized_control(self):
        nominal = np.array([0.9, 0.0])
        h, lg_h = cone_barrier_terms(0.55, 0.12, 0.25)
        control_zero, _, _, _ = self.controller._solve_socp(
            nominal, h, lg_h, epsilon_adapt=0.0
        )
        control_adaptive, _, _, _ = self.controller._solve_socp(
            nominal, h, lg_h, epsilon_adapt=0.12
        )
        self.assertGreater(
            np.linalg.norm(control_adaptive - control_zero), 1e-5
        )

    def test_checkpoint_round_trip_and_runtime_geometry_checks(self):
        checkpoint = {
            "format_version": CHECKPOINT_VERSION,
            "config": {
                "input_dim": 21,
                "hidden_sizes": [8],
                "ray_length": 3.0,
                "ray_angles": self.angles.tolist(),
                "d_safe": 0.25,
                "d_safe_multiplier": 3.0,
                "cbf_gain": 1.5,
                "slack_penalty": 100.0,
                "solver": "CLARABEL",
            },
            "model_state_dicts": [model.state_dict() for model in self.models],
            "calibration": {"phi_cal": np.array([2.0, 3.0])},
            "lipschitz": self.lipschitz,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atom.pt"
            torch.save(checkpoint, path)
            loaded = ATOMCBFController.from_checkpoint(
                path,
                max_movement_speed=1.0,
                max_turn_speed_degrees=92.0,
                expected_ray_length=3.0,
                expected_ray_angles=self.angles,
                expected_d_safe=0.25,
                expected_d_safe_multiplier=3.0,
                expected_cbf_gain=1.5,
            )
            np.testing.assert_allclose(loaded.phi_cal, [2.0, 3.0])
            with self.assertRaisesRegex(ValueError, "d_safe"):
                ATOMCBFController.from_checkpoint(
                    path,
                    max_movement_speed=1.0,
                    max_turn_speed_degrees=92.0,
                    expected_d_safe=0.3,
                )


if __name__ == "__main__":
    unittest.main()

