# Erwin Lejeune - 2026-02-15
"""Tests for Frenet optimal trajectory planner."""

import numpy as np

from uav_sim.trajectory_planning.frenet_optimal import FrenetOptimalPlanner


class TestFrenetOptimalPlanner:
    def _straight_ref(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [10.0, 0.0, 5.0],
                [15.0, 0.0, 5.0],
                [20.0, 0.0, 5.0],
            ]
        )

    def test_finds_feasible_trajectory(self):
        planner = FrenetOptimalPlanner(dt=0.2, horizon=3.0, n_lat_samples=5, n_speed_samples=3)
        best, cands = planner.plan(self._straight_ref())
        assert best is not None
        assert best.cost < float("inf")
        assert len(cands) > 0

    def test_respects_obstacle(self):
        planner = FrenetOptimalPlanner(
            max_lat_offset=2.0,
            n_lat_samples=5,
            n_speed_samples=3,
            dt=0.2,
            horizon=3.0,
        )
        obs = [(np.array([5.0, 0.0, 5.0]), 3.0)]
        best, _ = planner.plan(self._straight_ref(), obstacles=obs)
        if best is not None:
            dists = np.sqrt((best.x - 5.0) ** 2 + (best.y - 0.0) ** 2 + (best.z - 5.0) ** 2)
            assert np.all(dists >= 2.5)

    def test_zero_offset_cheapest(self):
        planner = FrenetOptimalPlanner(
            max_lat_offset=3.0,
            n_lat_samples=7,
            w_lateral=10.0,
            dt=0.2,
            horizon=3.0,
        )
        best, _ = planner.plan(self._straight_ref())
        assert best is not None
        assert abs(best.d[-1]) < 1.0

    def test_cartesian_output_length(self):
        planner = FrenetOptimalPlanner(dt=0.1, horizon=2.0)
        best, _ = planner.plan(self._straight_ref())
        assert best is not None
        expected_len = int(2.0 / 0.1) + 1
        assert len(best.x) == expected_len
