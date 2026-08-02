import math
import unittest

import numpy as np

from testing_utils import CBF_from_obs, cbf_velocity_filter_qp, generate_angles_rad


class CBFTests(unittest.TestCase):
    def test_ray_angles_match_sensor_layout(self):
        angles = generate_angles_rad(10, 90)
        self.assertEqual(len(angles), 21)
        self.assertAlmostEqual(angles[0], math.pi / 2)
        self.assertAlmostEqual(angles[10], 0.0)
        self.assertAlmostEqual(angles[-1], -math.pi / 2)

    def test_nominal_action_is_unchanged_without_nearby_obstacles(self):
        v_safe, omega_safe = cbf_velocity_filter_qp(
            0.7,
            0.2,
            np.array([3.0]),
            np.array([0.0]),
            d_safe=0.5,
            alpha=1.0,
            d_safe_threshold_mult=2.0,
            max_movement_speed=1.0,
            max_turn_speed=1.0,
        )
        self.assertAlmostEqual(v_safe, 0.7)
        self.assertAlmostEqual(omega_safe, 0.2)

    def test_frontal_obstacle_produces_feasible_deceleration(self):
        v_safe, omega_safe = cbf_velocity_filter_qp(
            1.0,
            0.0,
            np.array([0.6]),
            np.array([0.0]),
            d_safe=0.5,
            alpha=1.0,
            d_safe_threshold_mult=3.0,
            max_movement_speed=1.0,
            max_turn_speed=1.0,
        )

        # -2*x*v + alpha*(x^2-d_safe^2) >= 0
        barrier_value = -2.0 * 0.6 * v_safe + (0.6**2 - 0.5**2)
        self.assertGreaterEqual(barrier_value, -1e-5)
        self.assertGreater(v_safe, 0.0)
        self.assertLess(v_safe, 1.0)
        self.assertAlmostEqual(omega_safe, 0.0, places=5)

    def test_lateral_obstacle_turns_away_within_limits(self):
        _, omega_safe = cbf_velocity_filter_qp(
            0.5,
            0.0,
            np.array([0.4]),
            np.array([math.pi / 2]),
            d_safe=0.5,
            alpha=1.0,
            d_safe_threshold_mult=3.0,
            max_movement_speed=1.0,
            max_turn_speed=0.5,
        )
        self.assertLess(omega_safe, 0.0)
        self.assertGreaterEqual(omega_safe, -0.5)

    def test_wrapper_uses_physical_limits_and_returns_normalized_action(self):
        action = CBF_from_obs(
            ray_obs=np.array([0.2]),
            action=np.array([1.0, 0.0]),
            ray_original_lenght=3.0,
            max_movement_speed=1.0,
            max_turn_speed=92.0,
            d_safe=0.5,
            alpha=1.0,
            d_safe_mul=3.0,
            precomputed_angles_rad=np.array([0.0]),
        )
        self.assertGreaterEqual(action[0], 0.0)
        self.assertLess(action[0], 1.0)
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_mismatched_rays_are_rejected(self):
        with self.assertRaises(ValueError):
            cbf_velocity_filter_qp(
                0.5,
                0.0,
                np.array([0.5, 0.6]),
                np.array([0.0]),
            )


if __name__ == "__main__":
    unittest.main()
