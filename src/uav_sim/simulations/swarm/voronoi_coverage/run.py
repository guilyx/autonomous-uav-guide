# Erwin Lejeune - 2026-02-19
"""Voronoi coverage (Lloyd's algorithm): 100 m env, coverage areas shown.

6 agents converge to Voronoi centroids in a 100x100 XY region. The
Voronoi tessellation is drawn at each frame, showing each agent's
coverage area. Agents are rendered as quad models.

Reference: J. Cortes et al., "Coverage Control for Mobile Sensing Networks,"
IEEE T-RA, 2004.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

from uav_sim.swarm.coverage import CoverageController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 100.0
CRUISE_ALT = 50.0


def _clip_voronoi_region(vertices: np.ndarray, ws: float) -> np.ndarray:
    """Clip polygon vertices to [0, ws] box."""
    clipped = np.clip(vertices, 0, ws)
    return clipped


def main() -> None:
    n_ag = 6
    rng = np.random.default_rng(4)
    bounds = np.array([[0.0, WORLD_SIZE], [0.0, WORLD_SIZE]])
    pos_2d = rng.uniform(10, WORLD_SIZE - 10, (n_ag, 2))
    ctrl = CoverageController(bounds=bounds, resolution=2.0, gain=0.5)

    dt, n_steps = 0.1, 500

    snap: list[np.ndarray] = [pos_2d.copy()]
    coverage_cost = np.zeros(n_steps)
    mean_dist = np.zeros(n_steps)

    for step in range(n_steps):
        forces = ctrl.compute_forces(pos_2d)
        pos_2d = pos_2d + forces * dt
        pos_2d = np.clip(pos_2d, 1.0, WORLD_SIZE - 1.0)
        snap.append(pos_2d.copy())
        coverage_cost[step] = np.sum(np.linalg.norm(forces, axis=1))
        dists = [
            np.linalg.norm(pos_2d[i] - pos_2d[j]) for i in range(n_ag) for j in range(i + 1, n_ag)
        ]
        mean_dist[step] = np.mean(dists) if dists else 0.0

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.Set1(np.linspace(0, 1, n_ag))

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[0, 2])
    ax_cost = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1:])

    fig.suptitle("Voronoi Coverage (Lloyd's Algorithm, 100m env)", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_top.set_xlim(0, WORLD_SIZE)
    ax_top.set_ylim(0, WORLD_SIZE)
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.set_title("Top Down — Voronoi Regions", fontsize=9)
    ax_top.set_aspect("equal")

    ax_side.set_xlim(0, WORLD_SIZE)
    ax_side.set_ylim(0, WORLD_SIZE)
    ax_side.set_xlabel("X")
    ax_side.set_ylabel("Z")
    ax_side.set_title("Side View", fontsize=9)
    ax_side.grid(True, alpha=0.15)

    ax_cost.set_xlim(0, n_steps * dt)
    ax_cost.set_ylim(0, max(1.0, coverage_cost.max() * 1.1))
    ax_cost.set_xlabel("Time [s]", fontsize=8)
    ax_cost.set_ylabel("Coverage Force", fontsize=8)
    ax_cost.set_title("Coverage Cost", fontsize=9)
    ax_cost.grid(True, alpha=0.3)
    (lcost,) = ax_cost.plot([], [], "m-", lw=0.8)

    ax_dist.set_xlim(0, n_steps * dt)
    ax_dist.set_ylim(0, max(10, mean_dist.max() * 1.2))
    ax_dist.set_xlabel("Time [s]", fontsize=8)
    ax_dist.set_ylabel("Mean Neighbor Distance [m]", fontsize=8)
    ax_dist.set_title("Neighbor Distances", fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    (ldist,) = ax_dist.plot([], [], "b-", lw=0.8)

    trails_top = [ax_top.plot([], [], color=colors[i], lw=0.6, alpha=0.4)[0] for i in range(n_ag)]
    sc_top = ax_top.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=60, zorder=10)

    voronoi_patches: list = []
    veh_arts: list = []
    title = ax3d.set_title("t = 0.0 s")

    anim = SimAnimator("voronoi_coverage", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        k = idx[f]
        p = snap[k]

        sc_top.set_offsets(p)

        # Draw Voronoi regions
        for patch in voronoi_patches:
            patch.remove()
        voronoi_patches.clear()

        mirror = np.vstack(
            [
                p,
                np.column_stack([-p[:, 0], p[:, 1]]),
                np.column_stack([2 * WORLD_SIZE - p[:, 0], p[:, 1]]),
                np.column_stack([p[:, 0], -p[:, 1]]),
                np.column_stack([p[:, 0], 2 * WORLD_SIZE - p[:, 1]]),
            ]
        )
        try:
            vor = Voronoi(mirror)
            for i in range(n_ag):
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    verts = _clip_voronoi_region(vor.vertices[region], WORLD_SIZE)
                    poly = plt.Polygon(verts, alpha=0.15, fc=colors[i], ec=colors[i], lw=0.8)
                    ax_top.add_patch(poly)
                    voronoi_patches.append(poly)
        except Exception:
            pass

        for i in range(n_ag):
            hist = np.array([snap[j][i] for j in range(0, k + 1, max(1, k // 50))])
            trails_top[i].set_data(hist[:, 0], hist[:, 1])

        clear_vehicle_artists(veh_arts)
        for i in range(n_ag):
            R = Quadrotor.rotation_matrix(0, 0, 0)
            pos3 = np.array([p[i, 0], p[i, 1], CRUISE_ALT])
            veh_arts.extend(
                draw_quadrotor_3d(
                    ax3d, pos3, R, size=3.0, arm_colors=(colors[i][:3], colors[i][:3])
                )
            )

        lcost.set_data(times[:k], coverage_cost[:k])
        ldist.set_data(times[:k], mean_dist[:k])
        title.set_text(f"Voronoi Coverage — t = {k * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
