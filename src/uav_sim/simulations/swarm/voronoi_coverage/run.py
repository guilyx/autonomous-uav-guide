# Erwin Lejeune - 2026-02-18
"""Voronoi coverage (Lloyd's algorithm): 3D + top-down + cost panels.

Shows 6 agents converging to Voronoi centroids in a 30 m environment.
Drones hover at a fixed altitude while the coverage algorithm runs in XY.

Three panels:
  - 3D scene with quadrotor models at cruise altitude.
  - Top-down (XY) with coloured trails.
  - Coverage cost + displacement inset.

Reference: J. Cortes et al., "Coverage Control for Mobile Sensing Networks,"
IEEE T-RA, 2004.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.coverage import CoverageController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def main() -> None:
    n_ag = 6
    rng = np.random.default_rng(4)
    bounds = np.array([[0, WORLD_SIZE], [0, WORLD_SIZE]])
    pos_2d = rng.uniform(3, WORLD_SIZE - 3, (n_ag, 2))
    ctrl = CoverageController(bounds=bounds, resolution=1.0, gain=2.0)
    dt, n_steps = 0.05, 300

    snap: list[np.ndarray] = [pos_2d.copy()]
    coverage_cost = np.zeros(n_steps)
    for step in range(n_steps):
        forces = ctrl.compute_forces(pos_2d)
        pos_2d = pos_2d + forces * dt
        pos_2d = np.clip(pos_2d, 0, WORLD_SIZE)
        snap.append(pos_2d.copy())
        coverage_cost[step] = np.sum(np.linalg.norm(forces, axis=1))

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.Set1(np.linspace(0, 1, n_ag))

    # ── 3-Panel viz ────────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="Voronoi Coverage (Lloyd's Algorithm)", world_size=WORLD_SIZE, figsize=(18, 9)
    )

    anim = SimAnimator("voronoi_coverage", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    # Top-down trails
    trails_top = [
        viz.ax_top.plot([], [], color=colors[i], lw=0.8, alpha=0.5)[0] for i in range(n_ag)
    ]
    sc_top = viz.ax_top.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=60, zorder=5)

    # 3D trails
    trails_3d = [
        viz.ax3d.plot([], [], [], color=colors[i], lw=0.6, alpha=0.4)[0] for i in range(n_ag)
    ]

    # Inset: cost + displacement
    ax_cost = viz.fig.add_axes([0.58, 0.03, 0.38, 0.22])
    ax_cost.set_xlim(0, n_steps * dt)
    ax_cost.set_ylim(0, max(0.5, coverage_cost.max() * 1.1))
    ax_cost.set_xlabel("Time [s]", fontsize=7)
    ax_cost.set_ylabel("Force Mag.", fontsize=7)
    ax_cost.tick_params(labelsize=6)
    ax_cost.grid(True, alpha=0.2)
    (lcost,) = ax_cost.plot([], [], "m-", lw=0.8)

    disp = np.zeros(n_steps)
    for step in range(1, n_steps):
        disp[step] = np.mean(np.linalg.norm(snap[step] - snap[step - 1], axis=1))

    ax_disp = ax_cost.twinx()
    ax_disp.set_ylim(0, max(0.01, disp.max() * 1.2))
    ax_disp.set_ylabel("Displacement", fontsize=7, color="blue")
    ax_disp.tick_params(labelsize=6, colors="blue")
    (ldisp,) = ax_disp.plot([], [], "b--", lw=0.6)

    veh_arts: list = []

    def update(f: int) -> None:
        k = idx[f]
        p = snap[k]
        sc_top.set_offsets(p)

        for i in range(n_ag):
            hist = np.array([snap[j][i] for j in range(k + 1)])
            trails_top[i].set_data(hist[:, 0], hist[:, 1])
            trails_3d[i].set_data(hist[:, 0], hist[:, 1])
            trails_3d[i].set_3d_properties(np.full(len(hist), CRUISE_ALT))

        clear_vehicle_artists(veh_arts)
        for i in range(n_ag):
            R = Quadrotor.rotation_matrix(0, 0, 0)
            pos3 = np.array([p[i, 0], p[i, 1], CRUISE_ALT])
            veh_arts.extend(
                draw_quadrotor_3d(
                    viz.ax3d, pos3, R, size=1.2, arm_colors=(colors[i][:3], colors[i][:3])
                )
            )

        lcost.set_data(times[:k], coverage_cost[:k])
        ldisp.set_data(times[:k], disp[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
