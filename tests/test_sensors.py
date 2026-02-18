# Erwin Lejeune - 2026-02-18
"""Tests for sensor models."""

import numpy as np
import pytest

from uav_sim.sensors.base import SensorMount
from uav_sim.sensors.camera import Camera, CameraIntrinsics
from uav_sim.sensors.gps import GPS
from uav_sim.sensors.imu import IMU
from uav_sim.sensors.lidar import Lidar2D, Lidar3D
from uav_sim.sensors.range_finder import RangeFinder


class TestSensorMount:
    def test_default_identity(self):
        m = SensorMount()
        np.testing.assert_array_equal(m.position, np.zeros(3))
        R = m.rotation_matrix()
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_yaw_rotation(self):
        m = SensorMount(orientation=np.array([0.0, 0.0, np.pi / 2]))
        R = m.rotation_matrix()
        np.testing.assert_allclose(R @ np.array([1, 0, 0]), np.array([0, 1, 0]), atol=1e-10)


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

    def test_mount_attribute(self):
        imu = IMU(seed=42)
        assert imu.mount is not None
        np.testing.assert_array_equal(imu.mount.position, np.zeros(3))


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


class TestLidar2D:
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

    def test_custom_mount(self):
        mount = SensorMount(position=np.array([0.1, 0.0, -0.05]))
        lidar = Lidar2D(num_beams=10, seed=42, mount=mount)
        np.testing.assert_allclose(lidar.mount.position, [0.1, 0.0, -0.05])


class TestLidar3D:
    def test_output_shape(self):
        lidar = Lidar3D(num_beams_h=20, num_beams_v=4, seed=42)
        state = np.zeros(12)
        m = lidar.sense(state)
        assert m.shape == (4, 20)

    def test_max_range_no_obstacles(self):
        lidar = Lidar3D(num_beams_h=10, num_beams_v=3, max_range=15.0, noise_std=0.0, seed=42)
        state = np.zeros(12)
        m = lidar.sense(state)
        np.testing.assert_allclose(m, 15.0, atol=0.1)

    def test_to_point_cloud_empty(self):
        lidar = Lidar3D(num_beams_h=10, num_beams_v=3, max_range=15.0, noise_std=0.0, seed=42)
        state = np.zeros(12)
        lidar.sense(state)
        pc = lidar.to_point_cloud(state)
        assert pc.shape == (0, 3)

    def test_to_point_cloud_with_obstacle(self):
        from uav_sim.environment.obstacles import SphereObstacle
        from uav_sim.environment.world import World

        world = World(bounds_min=np.zeros(3), bounds_max=np.full(3, 20.0))
        world.add_obstacle(SphereObstacle(centre=np.array([5.0, 0.0, 0.0]), radius=1.0))
        lidar = Lidar3D(num_beams_h=36, num_beams_v=4, max_range=10.0, noise_std=0.0, seed=42)
        state = np.zeros(12)
        lidar.sense(state, world)
        pc = lidar.to_point_cloud(state)
        assert len(pc) > 0
        assert pc.shape[1] == 3


class TestCamera:
    def test_intrinsics(self):
        K = CameraIntrinsics()
        assert K.K.shape == (3, 3)

    def test_fov_properties(self):
        intrinsics = CameraIntrinsics(fx=320.0, fy=320.0, width=640, height=480)
        assert intrinsics.h_fov > 0
        assert intrinsics.v_fov > 0
        assert intrinsics.diagonal_fov > intrinsics.h_fov

    def test_project(self):
        cam = Camera()
        points = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
        pose = np.zeros(6)
        px = cam.project(points, pose)
        assert px.shape == (2, 2)

    def test_frustum_corners(self):
        cam = Camera(max_depth=10.0)
        pos = np.array([0.0, 0.0, 5.0])
        R = np.eye(3)
        corners = cam.frustum_corners(pos, R, depth=10.0)
        assert corners.shape == (4, 3)
        assert np.all(corners[:, 2] > 5.0)

    def test_custom_mount(self):
        mount = SensorMount(
            position=np.array([0.1, 0.0, -0.05]),
            orientation=np.array([0.0, -0.2, 0.0]),
        )
        cam = Camera(mount=mount)
        np.testing.assert_allclose(cam.mount.position, [0.1, 0.0, -0.05])
        R = cam.mount.rotation_matrix()
        assert R.shape == (3, 3)


class TestRangeFinder:
    def test_output(self):
        rf = RangeFinder(noise_std=0.0, seed=42)
        state = np.array([0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        m = rf.sense(state)
        assert m.shape == (1,)
        assert m[0] == pytest.approx(5.0, abs=0.1)
