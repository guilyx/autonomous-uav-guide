# Erwin Lejeune - 2026-02-15
"""RRT* 3D: 3-panel, two-phase visualisation.

Phase 1 — Algorithm: tree growing with progressive reveal (slow frames).
           Path smoothing is shown as a distinct step.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit along path -> land.

Reference: S. Karaman, E. Frazzoli, "Sampling-based Algorithms for Optimal
Motion Planning," IJRR, 2011. DOI: 10.1177/0278364911406761
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from uav_sim.environment import default_world
from uav_sim.path_planning.rrt_3d import RRTStar3D
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import smooth_path_3d
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
SEARCH_REPEAT = 3
N_TREE_GROW = 30
N_RAW_PAUSE = 12
N_SMOOTH_PAUSE = 12
N_FLY_FRAMES = 60


def _box_to_sphere(b):
    centre = (b.min_corner + b.max_corner) / 2
    half_xy = (b.max_corner[:2] - b.min_corner[:2]) / 2
    radius = float(np.linalg.norm(half_xy)) * 1.1
    return (centre, radius)


def main() -> None:
    world, buildings = default_world()
    sphere_obs = [_box_to_sphere(b) for b in buildings]

    start = np.array([2.0, 2.0, 15.0])
    goal = np.array([28.0, 28.0, 15.0])

    planner = RRTStar3D(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
        obstacles=sphere_obs,
        step_size=2.0,
        goal_radius=3.0,
        max_iter=5000,
        goal_bias=0.20,
        gamma=15.0,
    )
    path = planner.plan(start, goal, seed=42)
    if path is None:
        print("No path found!")
        return

    raw_path = np.array(path)
    smooth_path = smooth_path_3d(raw_path, epsilon=2.0, min_spacing=1.5)

    quad = Quadrotor()
    quad.reset(position=np.array([start[0], start[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    flight_states = fly_mission(
        quad,
        ctrl,
        smooth_path,
        cruise_alt=12.0,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.5,
        landing_duration=2.5,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    # ── Tree data ─────────────────────────────────────────────────────
    tree_nodes = np.array(planner.nodes)
    tree_parents = planner.parents
    tree_segs_3d, tree_segs_2d = [], []
    for i in range(1, len(tree_parents)):
        pi = tree_parents[i]
        if pi >= 0:
            tree_segs_3d.append([tree_nodes[pi], tree_nodes[i]])
            tree_segs_2d.append([tree_nodes[pi, :2], tree_nodes[i, :2]])

    # ── Frame schedule ────────────────────────────────────────────────
    n_grow_slow = N_TREE_GROW * SEARCH_REPEAT
    fly_step = max(1, len(flight_pos) // N_FLY_FRAMES)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    total = n_grow_slow + N_RAW_PAUSE + N_SMOOTH_PAUSE + len(fly_frames)

    viz = ThreePanelViz(title="RRT* 3D — Tree → Smooth → Flight", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.mark_start_goal(start, goal)

    tree_col_3d = None
    if tree_segs_3d:
        tree_col_3d = Line3DCollection(tree_segs_3d, colors="cyan", linewidths=0.3, alpha=0.0)
        viz.ax3d.add_collection3d(tree_col_3d)

    from matplotlib.collections import LineCollection

    tree_col_top = None
    if tree_segs_2d:
        tree_col_top = LineCollection(tree_segs_2d, colors="cyan", linewidths=0.3, alpha=0.0)
        viz.ax_top.add_collection(tree_col_top)

    (raw_3d,) = viz.ax3d.plot([], [], [], "b-", lw=1.5, alpha=0.0)
    (smooth_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.5, alpha=0.0, label="Smoothed")
    (raw_top,) = viz.ax_top.plot([], [], "b-", lw=1.0, alpha=0.0)
    (smooth_top,) = viz.ax_top.plot([], [], "lime", lw=2.0, alpha=0.0)

    fly_trail = viz.create_trail_artists()
    viz.ax3d.legend(fontsize=7, loc="upper left")

    ax_d = viz.setup_data_axes(title="RRT* Stats", ylabel="Nodes")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, len(tree_nodes) * 1.1)
    ax_d.set_xlabel("Progress", fontsize=7)
    (l_nodes,) = ax_d.plot([], [], "c-", lw=0.8, label="Tree Size")
    ax_d.text(
        0.02,
        0.92,
        f"Path: {len(raw_path)} → {len(smooth_path)} smooth",
        transform=ax_d.transAxes,
        fontsize=6,
    )
    ax_d.legend(fontsize=5, loc="lower right")

    title = viz.ax3d.set_title("RRT* Tree Growing")
    anim = SimAnimator("rrt_star_3d", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    n_segs = len(tree_segs_3d)

    def update(f: int) -> None:
        p1_end = n_grow_slow
        p2_end = p1_end + N_RAW_PAUSE
        p3_end = p2_end + N_SMOOTH_PAUSE

        if f < p1_end:
            gi = f // SEARCH_REPEAT
            frac = min(1.0, (gi + 1) / N_TREE_GROW)
            if tree_col_3d:
                tree_col_3d.set_alpha(frac * 0.4)
            if tree_col_top:
                tree_col_top.set_alpha(frac * 0.3)
            n_show = max(1, int(frac * len(tree_nodes)))
            l_nodes.set_data([frac], [n_show])
            title.set_text(
                f"RRT* Growing — {n_show}/{len(tree_nodes)} nodes, " f"{int(frac * n_segs)} edges"
            )
        elif f < p2_end:
            raw_3d.set_alpha(1.0)
            raw_3d.set_data(raw_path[:, 0], raw_path[:, 1])
            raw_3d.set_3d_properties(raw_path[:, 2])
            raw_top.set_alpha(1.0)
            raw_top.set_data(raw_path[:, 0], raw_path[:, 1])
            title.set_text(f"Raw RRT* Path ({len(raw_path)} nodes)")
        elif f < p3_end:
            smooth_3d.set_alpha(1.0)
            smooth_3d.set_data(smooth_path[:, 0], smooth_path[:, 1])
            smooth_3d.set_3d_properties(smooth_path[:, 2])
            smooth_top.set_alpha(1.0)
            smooth_top.set_data(smooth_path[:, 0], smooth_path[:, 1])
            raw_3d.set_alpha(0.2)
            raw_top.set_alpha(0.2)
            title.set_text("Smoothed Path (RDP + Resample)")
        else:
            fi = f - p3_end
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Quadrotor Following RRT* Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
