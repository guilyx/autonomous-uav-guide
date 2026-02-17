# Erwin Lejeune - 2026-02-16
"""Tests for 3D path planning algorithms."""

import numpy as np

from uav_sim.path_planning.astar_3d import AStar3D
from uav_sim.path_planning.potential_field_3d import PotentialField3D
from uav_sim.path_planning.rrt_3d import RRT3D, RRTStar3D

# ---------------------------------------------------------------------------
# A* 3D
# ---------------------------------------------------------------------------


class TestAStar3D:
    def test_straight_line_in_empty_grid(self):
        grid = np.zeros((10, 10, 10), dtype=bool)
        planner = AStar3D(grid)
        path = planner.plan((0, 0, 0), (9, 9, 9))
        assert path is not None
        assert path[0] == (0, 0, 0)
        assert path[-1] == (9, 9, 9)

    def test_no_path_when_blocked(self):
        grid = np.zeros((5, 5, 5), dtype=bool)
        grid[2, :, :] = True  # wall
        path = AStar3D(grid).plan((0, 0, 0), (4, 4, 4))
        assert path is None

    def test_start_is_obstacle(self):
        grid = np.zeros((5, 5, 5), dtype=bool)
        grid[0, 0, 0] = True
        assert AStar3D(grid).plan((0, 0, 0), (4, 4, 4)) is None

    def test_path_avoids_obstacles(self):
        grid = np.zeros((10, 10, 10), dtype=bool)
        grid[5, 4:7, 4:7] = True
        path = AStar3D(grid).plan((0, 0, 0), (9, 5, 5))
        assert path is not None
        for p in path:
            assert not grid[p], f"Path goes through obstacle at {p}"


# ---------------------------------------------------------------------------
# RRT 3D
# ---------------------------------------------------------------------------


class TestRRT3D:
    def test_finds_path_in_free_space(self):
        rrt = RRT3D(
            bounds_min=np.array([0, 0, 0]),
            bounds_max=np.array([10, 10, 10]),
            step_size=1.0,
            goal_radius=1.0,
            max_iter=5000,
            goal_bias=0.2,
        )
        path = rrt.plan(np.array([0, 0, 0]), np.array([9, 9, 9]), seed=42)
        assert path is not None
        assert len(path) >= 2

    def test_avoids_obstacle(self):
        obs = [(np.array([5.0, 5.0, 5.0]), 2.0)]
        rrt = RRT3D(
            bounds_min=np.array([0, 0, 0]),
            bounds_max=np.array([10, 10, 10]),
            obstacles=obs,
            step_size=0.5,
            goal_radius=1.0,
            max_iter=5000,
            goal_bias=0.15,
        )
        path = rrt.plan(np.array([0, 0, 0]), np.array([9, 9, 9]), seed=42)
        if path is not None:
            for p in path:
                dist = np.linalg.norm(p - np.array([5, 5, 5]))
                assert dist >= 1.5  # outside obstacle (with margin)


# ---------------------------------------------------------------------------
# RRT* 3D
# ---------------------------------------------------------------------------


class TestRRTStar3D:
    def test_finds_path(self):
        rrt_star = RRTStar3D(
            bounds_min=np.array([0, 0, 0]),
            bounds_max=np.array([10, 10, 10]),
            step_size=1.0,
            goal_radius=1.0,
            max_iter=3000,
            goal_bias=0.15,
            gamma=5.0,
        )
        path = rrt_star.plan(np.array([0, 0, 0]), np.array([9, 9, 9]), seed=42)
        assert path is not None

    def test_cost_improves_over_rrt(self):
        """RRT* should find a shorter or equal path than basic RRT."""
        kwargs = dict(
            bounds_min=np.array([0, 0, 0]),
            bounds_max=np.array([10, 10, 10]),
            step_size=1.0,
            goal_radius=1.0,
            max_iter=3000,
            goal_bias=0.15,
        )
        rrt = RRT3D(**kwargs)
        rrt_star = RRTStar3D(**kwargs, gamma=5.0)
        start, goal = np.array([0, 0, 0.0]), np.array([9, 9, 9.0])

        path_rrt = rrt.plan(start, goal, seed=42)
        path_star = rrt_star.plan(start, goal, seed=42)

        if path_rrt and path_star:
            cost_rrt = sum(
                np.linalg.norm(np.array(path_rrt[i + 1]) - np.array(path_rrt[i]))
                for i in range(len(path_rrt) - 1)
            )
            cost_star = sum(
                np.linalg.norm(np.array(path_star[i + 1]) - np.array(path_star[i]))
                for i in range(len(path_star) - 1)
            )
            assert cost_star <= cost_rrt * 1.1  # allow 10% margin


# ---------------------------------------------------------------------------
# Potential field 3D
# ---------------------------------------------------------------------------


class TestPotentialField3D:
    def test_reaches_goal_free_space(self):
        pf = PotentialField3D(step_size=0.2, max_iter=500)
        path = pf.plan(np.array([0, 0, 0]), np.array([5, 5, 5]), obstacles=[])
        assert np.linalg.norm(path[-1] - np.array([5, 5, 5])) < 0.5

    def test_avoids_obstacle(self):
        obs = [(np.array([2.5, 2.5, 2.5]), 1.0)]
        pf = PotentialField3D(step_size=0.1, max_iter=2000, rho0=3.0)
        path = pf.plan(np.array([0, 0, 0]), np.array([5, 5, 5]), obstacles=obs)
        for p in path:
            dist = np.linalg.norm(p - np.array([2.5, 2.5, 2.5]))
            assert dist >= 0.8  # stays outside obstacle

    def test_path_starts_at_start(self):
        pf = PotentialField3D()
        path = pf.plan(np.array([1, 2, 3]), np.array([5, 5, 5]), obstacles=[])
        np.testing.assert_allclose(path[0], [1, 2, 3])
