# Erwin Lejeune - 2026-02-17
"""Tests for the costmap system."""

import numpy as np
import pytest

from uav_sim.costmap.costmap import LayeredCostmap
from uav_sim.costmap.inflation_layer import InflationLayer
from uav_sim.costmap.occupancy_grid import OccupancyGrid
from uav_sim.costmap.social_layer import SocialLayer
from uav_sim.environment.obstacles import SphereObstacle
from uav_sim.environment.world import DynamicAgent, World


class TestOccupancyGrid:
    def test_shape_2d(self):
        g = OccupancyGrid(
            resolution=1.0, bounds_min=np.zeros(3), bounds_max=np.full(3, 5.0)
        )
        assert g.dimensions == 2
        assert g.shape[0] == 6
        assert g.shape[1] == 6

    def test_shape_3d(self):
        g = OccupancyGrid(
            resolution=1.0,
            bounds_min=np.zeros(3),
            bounds_max=np.full(3, 5.0),
            dimensions=3,
        )
        assert len(g.shape) == 3

    def test_set_and_query(self):
        g = OccupancyGrid(
            resolution=0.5, bounds_min=np.zeros(3), bounds_max=np.full(3, 5.0)
        )
        g.set_occupied(np.array([2.0, 2.0, 0.0]))
        assert g.is_occupied(np.array([2.0, 2.0, 0.0]))
        assert not g.is_occupied(np.array([0.0, 0.0, 0.0]))

    def test_from_world(self):
        w = World(bounds_min=np.zeros(3), bounds_max=np.full(3, 10.0))
        w.add_obstacle(SphereObstacle(centre=np.array([5.0, 5.0, 0.0]), radius=1.0))
        g = OccupancyGrid(
            resolution=1.0, bounds_min=np.zeros(3), bounds_max=np.full(3, 10.0)
        )
        g.from_world(w)
        assert g.is_occupied(np.array([5.0, 5.0, 0.0]))


class TestInflationLayer:
    def test_inflation(self):
        g = OccupancyGrid(
            resolution=0.5, bounds_min=np.zeros(3), bounds_max=np.full(3, 5.0)
        )
        g.set_occupied(np.array([2.5, 2.5, 0.0]))
        layer = InflationLayer(inflation_radius=2.0, cost_scaling=1.0)
        inflated = layer.apply(g)
        cell = g.world_to_cell(np.array([2.5, 2.5, 0.0]))
        assert inflated[cell] == pytest.approx(1.0)
        # Neighbouring cell should have decayed cost
        nearby = g.world_to_cell(np.array([3.0, 2.5, 0.0]))
        assert inflated[nearby] > 0.0


class TestSocialLayer:
    def test_stationary_agent_no_cost(self):
        g = OccupancyGrid(
            resolution=0.5, bounds_min=np.zeros(3), bounds_max=np.full(3, 10.0)
        )
        w = World()
        w.add_agent(
            DynamicAgent(position=np.array([5.0, 5.0, 0.0]), velocity=np.zeros(3))
        )
        layer = SocialLayer()
        cost = layer.apply(g, w)
        assert np.max(cost) == 0.0

    def test_moving_agent_creates_cost(self):
        g = OccupancyGrid(
            resolution=0.5, bounds_min=np.zeros(3), bounds_max=np.full(3, 10.0)
        )
        w = World()
        w.add_agent(
            DynamicAgent(
                position=np.array([5.0, 5.0, 0.0]), velocity=np.array([2.0, 0.0, 0.0])
            )
        )
        layer = SocialLayer()
        cost = layer.apply(g, w)
        assert np.max(cost) > 0.0


class TestLayeredCostmap:
    def test_composite(self):
        g = OccupancyGrid(
            resolution=1.0, bounds_min=np.zeros(3), bounds_max=np.full(3, 10.0)
        )
        g.set_occupied(np.array([5.0, 5.0, 0.0]))
        cm = LayeredCostmap(g, inflation=InflationLayer())
        cm.update()
        assert cm.cost_at(np.array([5.0, 5.0, 0.0])) == pytest.approx(1.0)
