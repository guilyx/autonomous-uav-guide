# Erwin Lejeune - 2026-02-16
"""Geometric SO(3) controller demonstration: hover and attitude recovery."""

from __future__ import annotations

import numpy as np

from quadrotor_sim.control.geometric_controller import GeometricController
from quadrotor_sim.models.quadrotor import Quadrotor
from quadrotor_sim.visualization import plot_state_history


def main() -> None:
    """Demo: start tilted, recover to hover at z=1 using geometric control."""
    quad = Quadrotor()
    quad.reset(
        position=np.array([0.0, 0.0, 1.0]),
        euler=np.array([0.3, -0.2, 0.0]),  # initial tilt
    )
    hover_force = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hover_force))

    ctrl = GeometricController()
    target = np.array([0.0, 0.0, 1.0])

    dt = 0.001
    duration = 5.0
    steps = int(duration / dt)
    times = np.zeros(steps)
    states = np.zeros((steps, 12))

    for i in range(steps):
        wrench = ctrl.compute(quad.state, target)
        quad.step(wrench, dt)
        times[i] = quad.time
        states[i] = quad.state

    print(f"Final euler angles: {np.rad2deg(quad.euler)}")
    print(f"Final position: {quad.position}")

    plot_state_history(times, states, title="Geometric Control: Attitude Recovery")


if __name__ == "__main__":
    main()
