# Erwin Lejeune - 2026-02-18
"""Tests for the gimbal model."""

import numpy as np

from uav_sim.sensors.gimbal import Gimbal


class TestGimbal:
    def test_initial_angles(self):
        g = Gimbal()
        assert g.pan == 0.0
        assert g.tilt == -np.pi / 4

    def test_look_at_directly_below(self):
        g = Gimbal()
        pan, tilt = g.look_at(
            np.array([5.0, 5.0, 10.0]),
            np.array([5.0, 5.0, 0.0]),
            yaw=0.0,
        )
        assert abs(tilt - (-np.pi / 2)) < 0.01

    def test_step_rate_limiting(self):
        g = Gimbal(max_rate=1.0)
        g.reset(pan=0.0, tilt=0.0)
        g.step(desired_pan=10.0, desired_tilt=-10.0, dt=0.1)
        assert abs(g.pan - 0.1) < 1e-6
        assert abs(g.tilt - (-0.1)) < 1e-6

    def test_frustum_corners_shape(self):
        g = Gimbal()
        g.reset(pan=0.0, tilt=-np.pi / 4)
        corners = g.frustum_corners_world(
            np.array([5.0, 5.0, 10.0]),
            h_fov=np.radians(60),
            v_fov=np.radians(45),
            depth=15.0,
            yaw=0.0,
        )
        assert corners.shape == (4, 3)

    def test_rotation_matrix_zero_looks_forward(self):
        g = Gimbal()
        g.reset(pan=0.0, tilt=0.0)
        R = g.rotation_matrix(yaw=0.0)
        optical_axis = R @ np.array([0, 0, 1])
        np.testing.assert_allclose(optical_axis, [1, 0, 0], atol=1e-10)

    def test_rotation_matrix_down(self):
        g = Gimbal()
        g.reset(pan=0.0, tilt=-np.pi / 2)
        R = g.rotation_matrix(yaw=0.0)
        optical_axis = R @ np.array([0, 0, 1])
        np.testing.assert_allclose(optical_axis, [0, 0, -1], atol=1e-10)

    def test_look_at_forward(self):
        g = Gimbal()
        pan, tilt = g.look_at(
            np.array([0.0, 0.0, 5.0]),
            np.array([10.0, 0.0, 5.0]),
            yaw=0.0,
        )
        assert abs(pan) < 0.01
        assert abs(tilt) < 0.01

    def test_joint_limits_clamped(self):
        g = Gimbal(max_rate=100.0)
        g.reset(pan=0.0, tilt=0.0)
        g.step(desired_pan=10.0, desired_tilt=-10.0, dt=10.0)
        assert g.pan <= g.limits.pan_max
        assert g.tilt >= g.limits.tilt_min
