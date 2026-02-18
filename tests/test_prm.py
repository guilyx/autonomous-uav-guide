# Erwin Lejeune - 2026-02-15
"""Tests for PRM 3D path planner."""

import numpy as np

from uav_sim.path_planning.prm_3d import PRM3D


class TestPRM3D:
    def test_finds_path_in_free_space(self):
        prm = PRM3D(
            bounds_min=np.array([0, 0, 0]),
            bounds_max=np.array([10, 10, 10]),
            n_samples=100,
            k_neighbours=8,
        )
        path = prm.plan(np.array([1, 1, 1]), np.array([9, 9, 9]), seed=42)
        assert path is not None
        assert len(path) >= 2
        np.testing.assert_allclose(path[0], [1, 1, 1])
        np.testing.assert_allclose(path[-1], [9, 9, 9])

    def test_avoids_obstacle(self):
        obs = [(np.array([5.0, 5.0, 5.0]), 2.0)]
        prm = PRM3D(
            bounds_min=np.array([0, 0, 0]),
            bounds_max=np.array([10, 10, 10]),
            obstacles=obs,
            n_samples=100,
            k_neighbours=8,
        )
        path = prm.plan(np.array([1, 1, 1]), np.array([9, 9, 9]), seed=42)
        if path is not None:
            for p in path[1:-1]:
                dist = np.linalg.norm(np.array(p) - np.array([5, 5, 5]))
                assert dist >= 1.8

    def test_build_then_query(self):
        prm = PRM3D(
            bounds_min=np.zeros(3),
            bounds_max=np.full(3, 10.0),
            n_samples=80,
        )
        prm.build(seed=7)
        assert len(prm.nodes) == 80
        path = prm.plan(np.array([1, 1, 1]), np.array([9, 9, 9]))
        assert path is not None

    def test_returns_none_when_blocked(self):
        obs = [(np.array([5.0, 5.0, 5.0]), 6.0)]
        prm = PRM3D(
            bounds_min=np.zeros(3),
            bounds_max=np.full(3, 10.0),
            obstacles=obs,
            n_samples=30,
            k_neighbours=5,
        )
        path = prm.plan(np.array([1, 5, 5]), np.array([9, 5, 5]), seed=42)
        assert path is None
