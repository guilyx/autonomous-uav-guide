# Erwin Lejeune - 2026-02-17
"""Tests for sensor models."""

import numpy as np
import pytest

from uav_sim.sensors.camera import Camera, CameraIntrinsics
from uav_sim.sensors.gps import GPS
from uav_sim.sensors.imu import IMU
from uav_sim.sensors.lidar import Lidar2D
from uav_sim.sensors.range_finder import RangeFinder


class TestIMU:
    def test_output_shape(self):
        imu = IMU(seed=42)
        state = np.zeros(12)
        m = imu.sense(state)
        assert m.shape == (6,)

    def test_noise(self):
        imu = IMU(accel_noise_std=0.0, gyro_noise_std=0.0, seed=42)
        state = np.zeros(12)
        state[6:9] = [1.0, 0.0, 0.0]
        m = imu.sense(state)
        assert m[0] != 0.0  # bias is still present


class TestGPS:
    def test_output_shape(self):
        gps = GPS(seed=42)
        state = np.array([1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        m = gps.sense(state)
        assert m.shape == (3,)

    def test_dropout(self):
        gps = GPS(dropout_prob=1.0, seed=42)
        state = np.zeros(12)
        m = gps.sense(state)
        assert np.all(np.isnan(m))


class TestLidar:
    def test_output_shape(self):
        lidar = Lidar2D(num_beams=36, seed=42)
        state = np.zeros(12)
        m = lidar.sense(state)
        assert m.shape == (36,)

    def test_max_range_no_obstacles(self):
        lidar = Lidar2D(num_beams=10, max_range=20.0, noise_std=0.0, seed=42)
        state = np.zeros(12)
        m = lidar.sense(state)
        np.testing.assert_allclose(m, 20.0, atol=0.1)


class TestCamera:
    def test_intrinsics(self):
        K = CameraIntrinsics()
        assert K.K.shape == (3, 3)

    def test_project(self):
        cam = Camera()
        points = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
        pose = np.zeros(6)
        px = cam.project(points, pose)
        assert px.shape == (2, 2)


class TestRangeFinder:
    def test_output(self):
        rf = RangeFinder(noise_std=0.0, seed=42)
        state = np.array([0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        m = rf.sense(state)
        assert m.shape == (1,)
        assert m[0] == pytest.approx(5.0, abs=0.1)
