# Erwin Lejeune - 2026-02-15
"""Tests for nonlinear MPC tracker."""

import numpy as np

from uav_sim.trajectory_tracking.nmpc import NMPCTracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


class TestNMPCTracker:
    def test_hover_at_target(self):
        nmpc = NMPCTracker(horizon=5, dt=0.02)
        state = np.zeros(12)
        state[2] = 5.0
        wrench = nmpc.compute(state, ref_pos=np.array([0, 0, 5.0]))
        assert wrench.shape == (4,)
        assert wrench[0] > 0

    def test_thrust_increases_below_target(self):
        nmpc = NMPCTracker(horizon=5, dt=0.02)
        state = np.zeros(12)
        state[2] = 3.0
        wrench = nmpc.compute(state, ref_pos=np.array([0, 0, 5.0]))
        hover_T = 1.5 * 9.81
        assert wrench[0] > hover_T * 0.5

    def test_closed_loop_z_convergence(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 3.0]))
        hover_f = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_f))

        nmpc = NMPCTracker(horizon=4, dt=0.02)
        target = np.array([0.0, 0.0, 5.0])
        dt = 0.01
        wrench = quad.hover_wrench()
        ctrl_counter = 0.0
        for _ in range(800):
            ctrl_counter += dt
            if ctrl_counter >= 0.02 - 1e-8:
                wrench = nmpc.compute(quad.state, target)
                ctrl_counter = 0.0
            quad.step(wrench, dt)
        assert abs(quad.position[2] - 5.0) < 2.5

    def test_reset(self):
        nmpc = NMPCTracker()
        nmpc._warm = np.ones(32)
        nmpc.reset()
        assert nmpc._warm is None
