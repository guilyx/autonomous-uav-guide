# Erwin Lejeune - 2026-02-17
"""RRT* 3D: two-phase visualisation.

Phase 1 — Algorithm: tree growing node by node, edges drawn incrementally.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit along path -> land.

Reference: S. Karaman, E. Frazzoli, "Sampling-based Algorithms for Optimal
Motion Planning," IJRR, 2011. DOI: 10.1177/0278364911406761
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_planning.rrt_3d import RRTStar3D
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import smooth_path_3d
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")


def main() -> None:
    obs = [
        (np.array([3.0, 3.0, 3.0]), 1.5),
        (np.array([7.0, 5.0, 4.0]), 1.0),
        (np.array([5.0, 8.0, 6.0]), 1.2),
    ]
    planner = RRTStar3D(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, 10),
        obstacles=obs,
        step_size=1.0,
        goal_radius=1.0,
        max_iter=1500,
        goal_bias=0.15,
        gamma=8.0,
    )
    start, goal = np.zeros(3), np.array([9.0, 9.0, 9.0])
    path = planner.plan(start, goal, seed=42)
    if path is None:
        print("No path found!")
        return

    tree_nodes = np.array(planner.nodes)
    tree_parents = planner.parents
    path_pts = np.array(path)

    # Smooth the raw RRT* path: prune redundant nodes, resample evenly
    flight_path = smooth_path_3d(path_pts, epsilon=0.8, min_spacing=0.5)

    # ── Phase 2: fly mission via pure pursuit ─────────────────────────────
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=2.0, waypoint_threshold=1.0, adaptive=True)
    flight_states = fly_mission(
        quad,
        ctrl,
        flight_path,
        cruise_alt=float(flight_path[0, 2]),
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.0,
        landing_duration=2.0,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    # ── Animation setup ───────────────────────────────────────────────────
    # Pre-draw the full tree (static); animate only path reveal + drone flight
    fly_step = max(1, len(flight_pos) // 80)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_reveal = 10  # frames to reveal tree + path
    n_ff = len(fly_frames)
    total = n_reveal + n_ff

    anim = SimAnimator("rrt_star_3d", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(10, 7))
    anim._fig = fig
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    for c, r in obs:
        anim.draw_sphere(ax, c, r)
    ax.scatter(*start, c="green", s=120, marker="^", label="Start", zorder=5)
    ax.scatter(*goal, c="red", s=120, marker="v", label="Goal", zorder=5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.legend(fontsize=7, loc="upper left")

    # Batch-draw tree edges (much faster than individual plot calls)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    tree_segs = []
    for i in range(1, len(tree_parents)):
        pi = tree_parents[i]
        if pi >= 0:
            tree_segs.append([tree_nodes[pi], tree_nodes[i]])
    if tree_segs:
        tree_col = Line3DCollection(tree_segs, colors="cyan", linewidths=0.3, alpha=0.0)
        ax.add_collection3d(tree_col)

    (path_line,) = ax.plot([], [], [], "b-", lw=2.5, alpha=0.0)
    (fly_trail,) = ax.plot([], [], [], "orange", lw=1.8)
    title = ax.set_title("Phase 1: RRT* Tree")
    vehicle_arts: list = []

    def update(f):
        clear_vehicle_artists(vehicle_arts)
        if f < n_reveal:
            # Gradually reveal the tree then show the path
            alpha = min(1.0, (f + 1) / (n_reveal * 0.6))
            if tree_segs:
                tree_col.set_alpha(alpha * 0.4)
            if f >= n_reveal - 2:
                path_line.set_alpha(1.0)
                path_line.set_data(path_pts[:, 0], path_pts[:, 1])
                path_line.set_3d_properties(path_pts[:, 2])
            pct = int(100 * (f + 1) / n_reveal)
            title.set_text(f"Phase 1: RRT* Tree — {pct}%")
        else:
            fi = f - n_reveal
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            fly_trail.set_data(flight_pos[:k, 0], flight_pos[:k, 1])
            fly_trail.set_3d_properties(flight_pos[:k, 2])
            R = Quadrotor.rotation_matrix(*flight_states[k, 3:6])
            vehicle_arts.extend(draw_quadrotor_3d(ax, flight_pos[k], R, size=0.6))
            title.set_text("Phase 2: Quadrotor Following RRT* Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
