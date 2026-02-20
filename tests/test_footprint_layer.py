# Erwin Lejeune - 2026-02-18
"""Tests for footprint-based costmap inflation."""

import numpy as np

from uav_sim.costmap.footprint_layer import FootprintInflationLayer
from uav_sim.costmap.occupancy_grid import OccupancyGrid
from uav_sim.vehicles.footprint import CircularFootprint, RectangularFootprint


class TestFootprintInflationLayer:
    def test_inflates_around_obstacles(self):
        grid = OccupancyGrid(
            resolution=0.5,
            bounds_min=np.zeros(3),
            bounds_max=np.full(3, 10.0),
        )
        grid.grid[10, 10] = 1.0  # single obstacle
        fp = CircularFootprint(radius=0.5)
        layer = FootprintInflationLayer(fp, padding=0.2)
        result = layer.apply(grid)
        assert result[10, 10] == 1.0
        assert result[11, 10] > 0.0  # inflated cell

    def test_radius_matches_footprint(self):
        fp = CircularFootprint(radius=0.4)
        layer = FootprintInflationLayer(fp, padding=0.1)
        assert layer.inflation_radius == 0.5

    def test_rectangular_footprint(self):
        fp = RectangularFootprint(length=1.0, width=0.5)
        layer = FootprintInflationLayer(fp, padding=0.0)
        assert layer.inflation_radius > 0.0
