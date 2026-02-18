# Erwin Lejeune - 2026-02-15
"""Tests for MPC controller."""

import numpy as np

from uav_sim.path_tracking.mpc_controller import MPCController


class TestMPCController:
    def test_hover_at_target(self):
        ctrl = MPCController(horizon=5, dt=0.02)
        state = np.zeros(12)
        state[2] = 5.0
        wrench = ctrl.compute(state, target_pos=np.array([0, 0, 5.0]))
        assert wrench.shape == (4,)
        assert wrench[0] > 0

    def test_thrust_direction(self):
        ctrl = MPCController(horizon=5, dt=0.02)
        state = np.zeros(12)
        state[2] = 3.0
        wrench = ctrl.compute(state, target_pos=np.array([0, 0, 5.0]))
        assert wrench[0] > 0.027 * 9.81

    def test_reset(self):
        ctrl = MPCController()
        ctrl._warm_start = np.ones(40)
        ctrl.reset()
        assert ctrl._warm_start is None

    def test_with_velocity_ref(self):
        ctrl = MPCController(horizon=5, dt=0.02)
        state = np.zeros(12)
        state[:3] = [1.0, 1.0, 5.0]
        wrench = ctrl.compute(
            state, target_pos=np.array([5.0, 5.0, 5.0]), target_vel=np.array([1.0, 1.0, 0.0])
        )
        assert wrench.shape == (4,)
