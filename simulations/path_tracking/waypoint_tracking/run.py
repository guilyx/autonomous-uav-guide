# Erwin Lejeune - 2026-02-17
"""3-D waypoint tracking: multi-panel live simulation.

Left panel: 3D trajectory with waypoint markers and next-WP indicator.
Right panels: position and tracking-error time histories.

Reference: T. Lee et al., "Geometric Tracking Control of a Quadrotor UAV on
SE(3)," CDC, 2010. DOI: 10.1109/CDC.2010.5717652
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_tracking.geometric_controller import GeometricController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")


def main() -> None:
    quad = Quadrotor()
    quad.reset(position=np.zeros(3))
    ctrl = GeometricController()
    wps = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1.5], [0, 1, 1], [0, 0, 1.0]])
    dt, hold = 0.002, 2.0
    hold_steps = int(hold / dt)

    states_list: list[np.ndarray] = []
    wp_indices: list[int] = []
    for wi, wp in enumerate(wps):
        for _ in range(hold_steps):
            quad.step(ctrl.compute(quad.state, wp), dt)
            states_list.append(quad.state.copy())
            wp_indices.append(wi)

    states = np.array(states_list)
    pos = states[:, :3]
    n = len(pos)
    times = np.arange(n) * dt
    wp_idx_arr = np.array(wp_indices)

    err = np.zeros(n)
    for i in range(n):
        err[i] = np.linalg.norm(pos[i] - wps[wp_idx_arr[i]])

    skip = max(1, n // 250)
    idx = list(range(0, n, skip))
    n_frames = len(idx)

    anim = SimAnimator("waypoint_tracking", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 7))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])
    fig.suptitle("Waypoint Tracking (Geometric Controller)", fontsize=13)

    ax3d.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c="r", s=80, marker="D", zorder=5)
    for i, wp in enumerate(wps):
        ax3d.text(wp[0], wp[1], wp[2] + 0.08, f"WP{i}", fontsize=7, ha="center")
    ax3d.set_xlim(-0.3, 1.3)
    ax3d.set_ylim(-0.3, 1.3)
    ax3d.set_zlim(-0.1, 1.8)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    (trail3d,) = ax3d.plot([], [], [], "b-", lw=1.2, alpha=0.6)
    (dot3d,) = ax3d.plot([], [], [], "ko", ms=6)
    (wp_marker,) = ax3d.plot([], [], [], "r*", ms=15)

    ax_pos.set_xlim(0, n * dt)
    ax_pos.set_ylim(-0.2, 1.8)
    ax_pos.set_ylabel("Pos [m]", fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    (lx,) = ax_pos.plot([], [], "r-", lw=1, label="x")
    (ly,) = ax_pos.plot([], [], "g-", lw=1, label="y")
    (lz,) = ax_pos.plot([], [], "b-", lw=1, label="z")
    ax_pos.legend(fontsize=6, ncol=3, loc="upper right")
    ax_pos.tick_params(labelsize=7)

    ax_err.set_xlim(0, n * dt)
    ax_err.set_ylim(0, max(0.5, err.max() * 1.1))
    ax_err.set_xlabel("Time [s]", fontsize=8)
    ax_err.set_ylabel("Tracking Error [m]", fontsize=8)
    ax_err.grid(True, alpha=0.3)
    (lerr,) = ax_err.plot([], [], "m-", lw=1)
    ax_err.tick_params(labelsize=7)

    title = ax3d.set_title("Waypoint Tracking")

    vehicle_arts: list = []

    def update(f):
        k = idx[f]
        trail3d.set_data(pos[:k, 0], pos[:k, 1])
        trail3d.set_3d_properties(pos[:k, 2])
        dot3d.set_data([pos[k, 0]], [pos[k, 1]])
        dot3d.set_3d_properties([pos[k, 2]])
        clear_vehicle_artists(vehicle_arts)
        R = Quadrotor.rotation_matrix(*states[k, 3:6])
        vehicle_arts.extend(draw_quadrotor_3d(ax3d, pos[k], R, scale=30.0))
        wi = wp_idx_arr[k]
        wp_marker.set_data([wps[wi, 0]], [wps[wi, 1]])
        wp_marker.set_3d_properties([wps[wi, 2]])
        title.set_text(f"Waypoint Tracking — WP {wi}/{len(wps) - 1}")
        lx.set_data(times[:k], pos[:k, 0])
        ly.set_data(times[:k], pos[:k, 1])
        lz.set_data(times[:k], pos[:k, 2])
        lerr.set_data(times[:k], err[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
