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

    def test_short_edge_through_an_obstacle_is_rejected(self):
        """An edge shorter than a few sample steps still has to be checked.

        Both endpoints of this 1.4 m edge clear the sphere by 0.1 m and the
        midpoint is 0.6 m inside it.  Sampling ``int(length / 0.5)`` points
        gives 2 for any edge up to 1.5 m, i.e. the endpoints and nothing
        else, so the roadmap accepted an edge straight through the obstacle.
        """
        prm = PRM3D(
            bounds_min=np.full(3, -5.0),
            bounds_max=np.full(3, 5.0),
            obstacles=[(np.array([0.0, 0.0, 0.0]), 0.6)],
        )
        a, b = np.array([-0.7, 0.0, 0.0]), np.array([0.7, 0.0, 0.0])

        assert prm._point_free(a) and prm._point_free(b)
        assert not prm._point_free(0.5 * (a + b))
        assert not prm._edge_free(a, b)

    def test_no_obstacle_wider_than_the_sample_step_slips_through(self):
        """A sphere bigger than the sampling step cannot hide between samples.

        Swept along edges of every awkward length, because the bug was
        length-dependent: it bit hardest just above 1 m and eased off as
        the edge grew long enough for the sample count to catch up.
        """
        radius = 0.3  # diameter 0.6 m, wider than the 0.5 m sample step
        for length in (0.6, 1.0, 1.4, 2.2, 2.9, 4.1, 7.3):
            for frac in np.linspace(0.1, 0.9, 9):
                prm = PRM3D(
                    bounds_min=np.full(3, -10.0),
                    bounds_max=np.full(3, 10.0),
                    obstacles=[(np.array([length * frac, 0.0, 0.0]), radius)],
                )
                a, b = np.zeros(3), np.array([length, 0.0, 0.0])
                if not (prm._point_free(a) and prm._point_free(b)):
                    continue
                assert not prm._edge_free(a, b), f"missed at length={length}, frac={frac}"
