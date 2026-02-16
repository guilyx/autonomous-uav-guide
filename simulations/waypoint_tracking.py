# Erwin Lejeune - 2026-02-16
"""3D waypoint tracking with geometric controller."""

from __future__ import annotations

import numpy as np

from quadrotor_sim.control.geometric_controller import GeometricController
from quadrotor_sim.models.quadrotor import Quadrotor
from quadrotor_sim.visualization import plot_state_history, plot_trajectory_3d


def main() -> None:
    """Follow a sequence of 3D waypoints using geometric control."""
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 0.0]))

    ctrl = GeometricController()

    waypoints = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    dt = 0.001
    hold_time = 2.0
    hold_steps = int(hold_time / dt)
    all_times: list[float] = []
    all_states: list[np.ndarray] = []

    for wp in waypoints:
        for _ in range(hold_steps):
            wrench = ctrl.compute(quad.state, wp)
            quad.step(wrench, dt)
            all_times.append(quad.time)
            all_states.append(quad.state.copy())

    positions = np.array([s[:3] for s in all_states])
    states_arr = np.array(all_states)
    times_arr = np.array(all_times)

    plot_trajectory_3d(positions, title="Geometric Control: Waypoint Tracking", reference=waypoints)
    plot_state_history(times_arr, states_arr, title="Geometric Control: State History")


if __name__ == "__main__":
    main()
