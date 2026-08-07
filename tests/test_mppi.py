import math
import unittest

import numpy as np

from mppi import MPPILocalPlanner
from testing_utils import generate_angles_rad


class MPPITests(unittest.TestCase):
    def _clear_scan(self, planner: MPPILocalPlanner) -> np.ndarray:
        return np.ones(planner.ray_angles.size, dtype=np.float64)

    def test_goal_to_the_right_turns_right(self):
        # Unity convention: positive angle_to_goal / positive omega = right.
        # No obstacles, so the only thing driving omega is the goal term.
        angles = generate_angles_rad(10, 90)
        planner = MPPILocalPlanner(
            ray_length=3.0,
            ray_angles=angles,
            max_movement_speed=1.0,
            max_turn_speed_degrees=92.0,
            num_samples=256,
            horizon=10,
            seed=0,
        )
        scan = self._clear_scan(planner)
        omegas = []
        for _ in range(5):
            result = planner.plan(scan, 0.0, 0.0, distance_to_goal=3.0, angle_to_goal_rad=math.radians(45))
            omegas.append(result.action[1])
        self.assertGreater(np.mean(omegas), 0.0)

    def test_goal_to_the_left_turns_left(self):
        angles = generate_angles_rad(10, 90)
        planner = MPPILocalPlanner(
            ray_length=3.0,
            ray_angles=angles,
            max_movement_speed=1.0,
            max_turn_speed_degrees=92.0,
            num_samples=256,
            horizon=10,
            seed=0,
        )
        scan = self._clear_scan(planner)
        omegas = []
        for _ in range(5):
            result = planner.plan(scan, 0.0, 0.0, distance_to_goal=3.0, angle_to_goal_rad=math.radians(-45))
            omegas.append(result.action[1])
        self.assertLess(np.mean(omegas), 0.0)

    def test_never_reverses(self):
        angles = generate_angles_rad(10, 90)
        planner = MPPILocalPlanner(
            ray_length=3.0,
            ray_angles=angles,
            max_movement_speed=1.0,
            max_turn_speed_degrees=92.0,
            num_samples=256,
            horizon=10,
            seed=1,
        )
        scan = self._clear_scan(planner)
        for _ in range(10):
            result = planner.plan(scan, 0.0, 0.0, distance_to_goal=3.0, angle_to_goal_rad=math.radians(170))
            self.assertGreaterEqual(result.action[0], 0.0)

    def test_frontal_obstacle_reduces_speed_toward_goal_straight_ahead(self):
        angles = generate_angles_rad(10, 90)
        planner = MPPILocalPlanner(
            ray_length=3.0,
            ray_angles=angles,
            max_movement_speed=1.0,
            max_turn_speed_degrees=92.0,
            num_samples=512,
            horizon=12,
            obstacle_radius=0.25,
            seed=2,
        )
        scan = self._clear_scan(planner)
        center = planner.ray_angles.size // 2
        # Wall ahead but still outside obstacle_radius, so there's room to
        # react over the horizon (a wall already inside obstacle_radius at
        # t=0 is an already-violated corner case, not what this checks).
        scan[center - 1 : center + 2] = 0.6 / planner.ray_length

        clear_planner = MPPILocalPlanner(
            ray_length=3.0,
            ray_angles=angles,
            max_movement_speed=1.0,
            max_turn_speed_degrees=92.0,
            num_samples=512,
            horizon=12,
            obstacle_radius=0.25,
            seed=2,
        )
        blocked_speeds = []
        clear_speeds = []
        for _ in range(5):
            blocked_speeds.append(
                planner.plan(scan, 0.0, 0.0, distance_to_goal=3.0, angle_to_goal_rad=0.0).action[0]
            )
            clear_speeds.append(
                clear_planner.plan(
                    self._clear_scan(clear_planner), 0.0, 0.0, distance_to_goal=3.0, angle_to_goal_rad=0.0
                ).action[0]
            )
        self.assertLess(np.mean(blocked_speeds), np.mean(clear_speeds))

    def test_mismatched_rays_are_rejected(self):
        angles = generate_angles_rad(10, 90)
        planner = MPPILocalPlanner(
            ray_length=3.0,
            ray_angles=angles,
            max_movement_speed=1.0,
            max_turn_speed_degrees=92.0,
        )
        with self.assertRaises(ValueError):
            planner.plan(
                np.ones(5), 0.0, 0.0, distance_to_goal=1.0, angle_to_goal_rad=0.0
            )


if __name__ == "__main__":
    unittest.main()
