# Erwin Lejeune - 2026-02-16
"""Minimum-snap trajectory through waypoints demonstration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from quadrotor_sim.planning.min_snap import MinSnapTrajectory


def main() -> None:
    """Generate and visualise a min-snap trajectory through 3D waypoints."""
    waypoints = np.array(
        [
            [0, 0, 0],
            [2, 0, 1],
            [4, 2, 1.5],
            [6, 2, 0.5],
            [8, 0, 0.0],
        ]
    )
    segment_times = np.array([1.5, 1.5, 1.5, 1.5])

    ms = MinSnapTrajectory()
    coeffs = ms.generate(waypoints, segment_times)
    times, positions = ms.evaluate(coeffs, segment_times, dt=0.01)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=2, label="Min-snap")
    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        color="red",
        s=80,
        zorder=5,
        label="Waypoints",
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Minimum-Snap Trajectory")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
