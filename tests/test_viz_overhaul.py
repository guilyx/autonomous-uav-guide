# Erwin Lejeune - 2026-02-15
"""Tests for the visualization overhaul: default_world, costmap_viz, four-panel."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from uav_sim.environment import default_world
from uav_sim.visualization.costmap_viz import (
    create_four_panel_figure,
    draw_costmap_heatmap,
    draw_costmap_surface,
    draw_occupancy_overlay,
)


class TestDefaultWorld:
    def test_returns_world_and_buildings(self):
        world, buildings = default_world()
        assert world.bounds_max[0] == 30.0
        assert len(buildings) > 0

    def test_deterministic_seed(self):
        _, b1 = default_world(seed=42)
        _, b2 = default_world(seed=42)
        for a, b in zip(b1, b2):
            np.testing.assert_array_equal(a.min_corner, b.min_corner)

    def test_custom_params(self):
        world, buildings = default_world(world_size=50.0, n_buildings=3, seed=99)
        assert world.bounds_max[0] == 50.0
        assert len(buildings) <= 3


class TestCostmapHeatmap:
    def test_returns_artists(self):
        fig, ax = plt.subplots()
        grid = np.random.rand(10, 10)
        arts = draw_costmap_heatmap(ax, grid, (0, 30, 0, 30), add_colorbar=False)
        assert len(arts) > 0
        plt.close(fig)

    def test_masked_free_space(self):
        fig, ax = plt.subplots()
        grid = np.zeros((10, 10))
        grid[5, 5] = 1.0
        arts = draw_costmap_heatmap(ax, grid, (0, 30, 0, 30), mask_free=True, add_colorbar=False)
        assert len(arts) > 0
        plt.close(fig)


class TestCostmapSurface:
    def test_returns_artists(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        grid = np.random.rand(10, 10)
        arts = draw_costmap_surface(ax, grid, (0, 30, 0, 30))
        assert len(arts) > 0
        plt.close(fig)


class TestOccupancyOverlay:
    def test_returns_artists(self):
        fig, ax = plt.subplots()
        occ = np.zeros((10, 10))
        occ[3:6, 3:6] = 1.0
        arts = draw_occupancy_overlay(ax, occ, (0, 30, 0, 30))
        assert len(arts) > 0
        plt.close(fig)


class TestFourPanel:
    def test_returns_four_axes(self):
        fig, ax3d, ax_top, ax_side, ax_sensor = create_four_panel_figure()
        assert ax3d is not None
        assert ax_top is not None
        assert ax_side is not None
        assert ax_sensor is not None
        plt.close(fig)
