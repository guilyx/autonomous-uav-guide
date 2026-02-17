# Erwin Lejeune - 2026-02-16
"""Tests for trajectory tracking algorithms."""

import numpy as np
import pytest

from quadrotor_sim.models.quadrotor import Quadrotor
from quadrotor_sim.tracking.feedback_linearisation import FeedbackLinearisationTracker
from quadrotor_sim.tracking.mppi import MPPITracker

# ---------------------------------------------------------------------------
# Feedback linearisation
# ---------------------------------------------------------------------------


class TestFeedbackLinearisation:
    def test_hover_thrust(self):
        tracker = FeedbackLinearisationTracker()
        state = np.zeros(12)
        state[2] = 1.0
        wrench = tracker.compute(state, ref_pos=np.array([0.0, 0.0, 1.0]))
        expected_T = tracker.mass * tracker.gravity
        assert pytest.approx(wrench[0], rel=0.2) == expected_T

    def test_returns_four_element_wrench(self):
        tracker = FeedbackLinearisationTracker()
        wrench = tracker.compute(np.zeros(12), np.zeros(3))
        assert wrench.shape == (4,)

    def test_closed_loop_hover(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.5]))
        hover_force = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_force))

        tracker = FeedbackLinearisationTracker()
        ref = np.array([0.0, 0.0, 1.0])

        dt = 0.001
        for _ in range(5000):
            wrench = tracker.compute(quad.state, ref)
            quad.step(wrench, dt)

        assert abs(quad.position[2] - 1.0) < 0.15


# ---------------------------------------------------------------------------
# MPPI
# ---------------------------------------------------------------------------


def _simple_dynamics(x, u, dt):
    """Simple 2D double integrator: x=[px, py, vx, vy]."""
    return np.array(
        [
            x[0] + x[2] * dt,
            x[1] + x[3] * dt,
            x[2] + u[0] * dt,
            x[3] + u[1] * dt,
        ]
    )


def _simple_cost(x, u, ref):
    """Quadratic cost to target."""
    if ref is None:
        ref = np.zeros(4)
    return float(np.sum((x[:2] - ref[:2]) ** 2) + 0.1 * np.sum(u**2))


class TestMPPI:
    def test_returns_control_of_correct_dim(self):
        mppi = MPPITracker(
            state_dim=4,
            control_dim=2,
            horizon=10,
            num_samples=50,
            dynamics=_simple_dynamics,
            cost_fn=_simple_cost,
            dt=0.1,
        )
        u = mppi.compute(
            np.array([0, 0, 0, 0.0]), reference=np.array([1, 1, 0, 0.0]), seed=42
        )
        assert u.shape == (2,)

    def test_drives_towards_target(self):
        mppi = MPPITracker(
            state_dim=4,
            control_dim=2,
            horizon=15,
            num_samples=200,
            lambda_=1.0,
            dynamics=_simple_dynamics,
            cost_fn=_simple_cost,
            control_std=2.0,
            dt=0.1,
        )
        state = np.array([0.0, 0.0, 0.0, 0.0])
        target = np.array([5.0, 5.0, 0.0, 0.0])

        for _ in range(50):
            u = mppi.compute(state, reference=target, seed=None)
            state = _simple_dynamics(state, u, 0.1)

        dist = np.linalg.norm(state[:2] - target[:2])
        assert dist < 3.0  # making progress towards target

    def test_reset_clears_sequence(self):
        mppi = MPPITracker(
            state_dim=4,
            control_dim=2,
            dynamics=_simple_dynamics,
            cost_fn=_simple_cost,
        )
        mppi.U[0] = [1.0, 2.0]
        mppi.reset()
        np.testing.assert_array_equal(mppi.U, 0.0)

    def test_raises_without_dynamics(self):
        mppi = MPPITracker(state_dim=4, control_dim=2)
        with pytest.raises(RuntimeError):
            mppi.compute(np.zeros(4))
