# Erwin Lejeune - 2026-02-15
"""Tests for the coordinate frame transform module."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from uav_sim.frames import (
    FrameID,
    batch_body_to_world,
    batch_world_to_body,
    body_to_sensor,
    body_to_world,
    euler_to_rotation,
    sensor_to_body,
    sensor_to_world,
    world_to_body,
    world_to_sensor,
)


class TestEulerToRotation:
    def test_identity(self):
        R = euler_to_rotation(0.0, 0.0, 0.0)
        assert_allclose(R, np.eye(3), atol=1e-12)

    def test_yaw_90(self):
        R = euler_to_rotation(0.0, 0.0, np.pi / 2)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        assert_allclose(R, expected, atol=1e-12)

    def test_orthonormal(self):
        R = euler_to_rotation(0.3, -0.1, 1.2)
        assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


class TestBodyWorld:
    def test_round_trip(self):
        pos = np.array([5.0, 3.0, 10.0])
        euler = np.array([0.1, -0.2, 0.8])
        pt = np.array([1.0, 2.0, 3.0])
        world_pt = body_to_world(pt, pos, euler)
        recovered = world_to_body(world_pt, pos, euler)
        assert_allclose(recovered, pt, atol=1e-10)

    def test_identity_transform(self):
        pos = np.array([1.0, 2.0, 3.0])
        euler = np.zeros(3)
        pt = np.array([4.0, 5.0, 6.0])
        assert_allclose(body_to_world(pt, pos, euler), pt + pos, atol=1e-12)

    def test_origin_maps_to_position(self):
        pos = np.array([10.0, 20.0, 5.0])
        euler = np.array([0.5, 0.3, -0.7])
        assert_allclose(body_to_world(np.zeros(3), pos, euler), pos, atol=1e-12)


class TestSensorBody:
    def test_round_trip(self):
        mount_pos = np.array([0.1, 0.0, -0.05])
        mount_ori = np.array([0.0, -np.pi / 4, 0.0])
        pt = np.array([1.0, 0.0, 0.0])
        body_pt = sensor_to_body(pt, mount_pos, mount_ori)
        recovered = body_to_sensor(body_pt, mount_pos, mount_ori)
        assert_allclose(recovered, pt, atol=1e-10)

    def test_zero_mount(self):
        mount_pos = np.zeros(3)
        mount_ori = np.zeros(3)
        pt = np.array([1.0, 2.0, 3.0])
        assert_allclose(sensor_to_body(pt, mount_pos, mount_ori), pt, atol=1e-12)


class TestFullChain:
    def test_sensor_world_round_trip(self):
        mount_pos = np.array([0.1, 0.0, -0.05])
        mount_ori = np.array([0.0, 0.0, np.pi / 6])
        veh_pos = np.array([15.0, 15.0, 12.0])
        veh_euler = np.array([0.02, -0.01, 1.5])
        pt_sensor = np.array([3.0, 0.0, 0.0])
        pt_world = sensor_to_world(pt_sensor, mount_pos, mount_ori, veh_pos, veh_euler)
        recovered = world_to_sensor(pt_world, mount_pos, mount_ori, veh_pos, veh_euler)
        assert_allclose(recovered, pt_sensor, atol=1e-10)


class TestBatch:
    def test_batch_matches_single(self):
        pos = np.array([5.0, 3.0, 10.0])
        euler = np.array([0.1, -0.2, 0.8])
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])
        batch_result = batch_body_to_world(pts, pos, euler)
        for i, pt in enumerate(pts):
            single = body_to_world(pt, pos, euler)
            assert_allclose(batch_result[i], single, atol=1e-10)

    def test_batch_round_trip(self):
        pos = np.array([5.0, 3.0, 10.0])
        euler = np.array([0.1, -0.2, 0.8])
        pts = np.random.default_rng(42).uniform(-5, 5, (20, 3))
        world_pts = batch_body_to_world(pts, pos, euler)
        recovered = batch_world_to_body(world_pts, pos, euler)
        assert_allclose(recovered, pts, atol=1e-10)


class TestFrameID:
    def test_enum_values(self):
        assert FrameID.WORLD != FrameID.BODY
        assert FrameID.BODY != FrameID.SENSOR
