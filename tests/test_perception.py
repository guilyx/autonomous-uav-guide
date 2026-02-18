# Erwin Lejeune - 2026-02-17
"""Tests for perception algorithms."""

import numpy as np

from uav_sim.perception.obstacle_detection import RangeObstacleDetector
from uav_sim.perception.point_cloud import (
    ranges_to_point_cloud,
    remove_ground,
    voxel_downsample,
)


class TestRangeObstacleDetector:
    def test_detect_cluster(self):
        det = RangeObstacleDetector(threshold=5.0, min_cluster_size=2)
        angles = np.linspace(-np.pi, np.pi, 36)
        ranges = np.full(36, 10.0)
        ranges[10:15] = 3.0  # obstacle cluster
        centroids = det.detect(ranges, angles)
        assert len(centroids) >= 1

    def test_no_obstacles(self):
        det = RangeObstacleDetector(threshold=5.0)
        ranges = np.full(36, 10.0)
        angles = np.linspace(-np.pi, np.pi, 36)
        assert len(det.detect(ranges, angles)) == 0


class TestPointCloud:
    def test_ranges_to_cloud(self):
        pos = np.array([0.0, 0.0, 1.0])
        yaw = 0.0
        angles = np.array([0.0, np.pi / 2])
        ranges = np.array([5.0, 3.0])
        cloud = ranges_to_point_cloud(pos, yaw, ranges, angles)
        assert cloud.shape == (2, 3)
        assert cloud[0, 0] > 0  # forward

    def test_voxel_downsample(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [5.0, 5.0, 5.0]])
        down = voxel_downsample(pts, voxel_size=0.5)
        assert len(down) <= 3

    def test_remove_ground(self):
        pts = np.array([[0, 0, 0.1], [0, 0, 1.0], [0, 0, 5.0]])
        filtered = remove_ground(pts, z_threshold=0.5)
        assert len(filtered) == 2
