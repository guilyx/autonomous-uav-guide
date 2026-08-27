# Erwin Lejeune - 2026-02-16
"""3D visualisation helpers for quadrotor simulations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray


def plot_quadrotor_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    arm_length: float = 0.0397,
    scale: float = 10.0,
) -> None:
    """Draw a quadrotor frame on a 3D axis.

    Args:
        ax: Matplotlib 3D axis.
        position: [x, y, z] in world frame.
        R: 3x3 rotation matrix (body → world).
        arm_length: Motor arm length [m].
        scale: Visual scale multiplier.
    """
    L = arm_length * scale
    s = L / np.sqrt(2.0)

    # Motor positions in body frame (X-configuration).
    motors_body = np.array(
        [
            [s, s, 0],
            [s, -s, 0],
            [-s, -s, 0],
            [-s, s, 0],
        ]
    ).T

    motors_world = R @ motors_body + position.reshape(3, 1)

    # Draw arms.
    colors = ["r", "b", "r", "b"]
    for i in range(4):
        ax.plot(
            [position[0], motors_world[0, i]],
            [position[1], motors_world[1, i]],
            [position[2], motors_world[2, i]],
            color=colors[i],
            linewidth=2,
        )

    # Draw motor circles.
    for i in range(4):
        ax.scatter(*motors_world[:, i], color=colors[i], s=20)

    # Draw centre of mass.
    ax.scatter(*position, color="k", s=30, marker="o")


def plot_trajectory_3d(
    positions: NDArray[np.floating],
    title: str = "Quadrotor Trajectory",
    reference: NDArray[np.floating] | None = None,
    show: bool = True,
) -> tuple[plt.Figure, Axes3D]:
    """Plot a 3D trajectory and optionally a reference path.

    Args:
        positions: (N, 3) array of positions.
        title: Plot title.
        reference: Optional (M, 3) reference trajectory.
        show: Whether to call plt.show().

    Returns:
        (fig, ax) tuple.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "b-",
        label="Actual",
        linewidth=1.5,
    )

    if reference is not None:
        ax.plot(
            reference[:, 0],
            reference[:, 1],
            reference[:, 2],
            "r--",
            label="Reference",
            linewidth=1.5,
        )

    ax.scatter(*positions[0, :3], color="g", s=80, marker="^", label="Start")
    ax.scatter(*positions[-1, :3], color="r", s=80, marker="v", label="End")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()

    if show:
        plt.show()

    return fig, ax


def plot_state_history(
    times: NDArray[np.floating],
    states: NDArray[np.floating],
    title: str = "State History",
    show: bool = True,
) -> plt.Figure:
    """Plot full 12-state history as 4 subplots.

    Args:
        times: (N,) time array.
        states: (N, 12) state array.
        title: Figure title.
        show: Whether to call plt.show().

    Returns:
        Figure object.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title)

    labels_groups = [
        (["x", "y", "z"], "Position [m]"),
        (["phi", "theta", "psi"], "Euler [rad]"),
        (["vx", "vy", "vz"], "Velocity [m/s]"),
        (["p", "q", "r"], "Angular rate [rad/s]"),
    ]

    for i, (labels, ylabel) in enumerate(labels_groups):
        for j, lbl in enumerate(labels):
            axes[i].plot(times, states[:, i * 3 + j], label=lbl)
        axes[i].set_ylabel(ylabel)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()

    if show:
        plt.show()

    return fig
