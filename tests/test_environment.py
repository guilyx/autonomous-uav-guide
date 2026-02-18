# Erwin Lejeune - 2026-02-17
"""Tests for the environment system: obstacles, world, buildings."""

import numpy as np
import pytest

from uav_sim.environment.buildings import add_city_grid
from uav_sim.environment.obstacles import BoxObstacle, CylinderObstacle, SphereObstacle
from uav_sim.environment.world import DynamicAgent, World, WorldType


class TestSphereObstacle:
    def test_contains(self):
        s = SphereObstacle(centre=np.array([5.0, 5.0, 5.0]), radius=2.0)
        assert s.contains(np.array([5.0, 5.0, 5.0]))
        assert not s.contains(np.array([10.0, 10.0, 10.0]))

    def test_distance(self):
        s = SphereObstacle(centre=np.array([0.0, 0.0, 0.0]), radius=1.0)
        assert s.distance(np.array([3.0, 0.0, 0.0])) == pytest.approx(2.0)
        assert s.distance(np.array([0.5, 0.0, 0.0])) == pytest.approx(-0.5)

    def test_bounding_box(self):
        s = SphereObstacle(centre=np.array([1.0, 2.0, 3.0]), radius=0.5)
        lo, hi = s.bounding_box()
        np.testing.assert_allclose(lo, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(hi, [1.5, 2.5, 3.5])


class TestBoxObstacle:
    def test_contains(self):
        b = BoxObstacle(min_corner=np.zeros(3), max_corner=np.ones(3))
        assert b.contains(np.array([0.5, 0.5, 0.5]))
        assert not b.contains(np.array([2.0, 0.5, 0.5]))

    def test_distance(self):
        b = BoxObstacle(min_corner=np.zeros(3), max_corner=np.ones(3))
        assert b.distance(np.array([0.5, 0.5, 0.5])) == pytest.approx(0.0)
        assert b.distance(np.array([2.0, 0.5, 0.5])) == pytest.approx(1.0)


class TestCylinderObstacle:
    def test_contains(self):
        c = CylinderObstacle(centre=np.zeros(3), radius=1.0, height=5.0)
        assert c.contains(np.array([0.0, 0.0, 2.5]))
        assert not c.contains(np.array([0.0, 0.0, 6.0]))

    def test_distance(self):
        c = CylinderObstacle(centre=np.zeros(3), radius=1.0, height=5.0)
        assert c.distance(np.array([2.0, 0.0, 2.5])) == pytest.approx(1.0)


class TestWorld:
    def test_in_bounds(self):
        w = World(bounds_min=np.zeros(3), bounds_max=np.full(3, 10.0))
        assert w.in_bounds(np.array([5.0, 5.0, 5.0]))
        assert not w.in_bounds(np.array([11.0, 5.0, 5.0]))

    def test_is_free(self):
        w = World(bounds_min=np.zeros(3), bounds_max=np.full(3, 10.0))
        w.add_obstacle(SphereObstacle(centre=np.array([5.0, 5.0, 5.0]), radius=1.0))
        assert w.is_free(np.array([0.0, 0.0, 0.0]))
        assert not w.is_free(np.array([5.0, 5.0, 5.0]))

    def test_nearest_obstacle_distance(self):
        w = World(bounds_min=np.zeros(3), bounds_max=np.full(3, 10.0))
        w.add_obstacle(SphereObstacle(centre=np.array([5.0, 5.0, 5.0]), radius=1.0))
        d = w.nearest_obstacle_distance(np.array([8.0, 5.0, 5.0]))
        assert d == pytest.approx(2.0)

    def test_world_type(self):
        w = World(world_type=WorldType.INDOOR)
        assert w.world_type == WorldType.INDOOR


class TestDynamicAgent:
    def test_step(self):
        a = DynamicAgent(
            position=np.zeros(3),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        a.step(1.0)
        np.testing.assert_allclose(a.position, [1.0, 0.0, 0.0])

    def test_world_agents(self):
        w = World()
        a = DynamicAgent(position=np.zeros(3), velocity=np.array([1.0, 0.0, 0.0]))
        w.add_agent(a)
        w.step_agents(2.0)
        np.testing.assert_allclose(a.position, [2.0, 0.0, 0.0])


class TestBuildings:
    def test_city_grid(self):
        w = World(bounds_max=np.full(3, 100.0))
        buildings = add_city_grid(w, n_blocks=(2, 2), seed=42)
        assert len(buildings) == 4
        assert len(w.obstacles) == 4
