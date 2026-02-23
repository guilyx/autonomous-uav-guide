# Erwin Lejeune - 2026-02-15
"""PRM 3D: 3-panel, two-phase visualisation.

Phase 1 — Algorithm: roadmap construction with progressive edge reveal,
           then Dijkstra query with path highlight.  Search frames are
           tripled to slow down the algorithm phase.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit along smoothed path.

Reference: L. E. Kavraki et al., "Probabilistic Roadmaps for Path Planning
in High-Dimensional Configuration Spaces," IEEE T-RA, 1996.
DOI: 10.1109/70.508439
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from uav_sim.costmap import LayeredCostmap
from uav_sim.environment import default_world
from uav_sim.logging import SimLogger
from uav_sim.path_planning.prm_3d import PRM3D
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
N_SAMPLE = 20
N_EDGE = 15
N_SEARCH = 10
N_SMOOTH = 10
N_FLY_FRAMES = 60


def _box_to_sphere(b):
    centre = (b.min_corner + b.max_corner) / 2
    half_xy = (b.max_corner[:2] - b.min_corner[:2]) / 2
    radius = float(np.linalg.norm(half_xy)) * 1.1
    return (centre, radius)


def main() -> None:
    world, buildings = default_world()
    sphere_obs = [_box_to_sphere(b) for b in buildings]
    costmap = LayeredCostmap.from_obstacles(
        buildings, world_size=WORLD_SIZE, resolution=0.5, inflation_radius=2.0
    )

    start = np.array([2.0, 2.0, 15.0])
    goal = np.array([28.0, 28.0, 15.0])

    planner = PRM3D(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
        obstacles=sphere_obs,
        n_samples=600,
        k_neighbours=15,
    )
    planner.build(seed=42)
    path = planner.plan(start, goal)
    if path is None:
        print("No path found!")
        return

    raw_path = np.array(path)
    smooth = smooth_path_3d(raw_path, epsilon=2.0, min_spacing=1.5)

    quad = Quadrotor()
    quad.reset(position=np.array([start[0], start[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    flight_states = fly_mission(
        quad,
        ctrl,
        smooth,
        cruise_alt=12.0,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.5,
        landing_duration=2.5,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    logger = SimLogger("prm_3d", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "PRM")
    logger.log_metadata("roadmap_nodes", len(planner.nodes))
    logger.log_metadata("raw_path_length", len(raw_path))
    logger.log_metadata("smooth_path_length", len(smooth))
    raw_len = float(np.sum(np.linalg.norm(np.diff(raw_path, axis=0), axis=1)))
    smooth_len = float(np.sum(np.linalg.norm(np.diff(smooth, axis=0), axis=1)))
    logger.log_summary("raw_path_m", raw_len)
    logger.log_summary("smooth_path_m", smooth_len)
    logger.save()

    # ── Edge collections ──────────────────────────────────────────────
    edge_segs_3d, edge_segs_2d = [], []
    for i, adj in enumerate(planner.edges):
        for j, _ in adj:
            if j > i:
                edge_segs_3d.append([planner.nodes[i], planner.nodes[j]])
                edge_segs_2d.append([planner.nodes[i, :2], planner.nodes[j, :2]])

    # ── Frame schedule ────────────────────────────────────────────────
    algo_frames = (N_SAMPLE + N_EDGE + N_SEARCH) * SEARCH_REPEAT
    fly_step = max(1, len(flight_pos) // N_FLY_FRAMES)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    total = algo_frames + N_SMOOTH * SEARCH_REPEAT + len(fly_frames)

    viz = ThreePanelViz(title="PRM 3D — Roadmap → Smooth → Flight", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    costmap.grid.visualize_2d(viz.ax_top, alpha=0.35)
    viz.mark_start_goal(start, goal)

    edge_col_3d = None
    if edge_segs_3d:
        edge_col_3d = Line3DCollection(edge_segs_3d, colors="cyan", linewidths=0.3, alpha=0.0)
        viz.ax3d.add_collection3d(edge_col_3d)

    edge_col_top = None
    if edge_segs_2d:
        edge_col_top = LineCollection(edge_segs_2d, colors="cyan", linewidths=0.3, alpha=0.0)
        viz.ax_top.add_collection(edge_col_top)

    node_scat_3d = viz.ax3d.scatter(
        planner.nodes[:, 0],
        planner.nodes[:, 1],
        planner.nodes[:, 2],
        c="cyan",
        s=2,
        alpha=0.0,
        zorder=3,
    )
    node_scat_top = viz.ax_top.scatter(
        planner.nodes[:, 0],
        planner.nodes[:, 1],
        c="cyan",
        s=1,
        alpha=0.0,
        zorder=3,
    )

    (raw_3d,) = viz.ax3d.plot([], [], [], "b-", lw=1.5, alpha=0.0)
    (smooth_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.5, alpha=0.0, label="Smoothed")
    (raw_top,) = viz.ax_top.plot([], [], "b-", lw=1.0, alpha=0.0)
    (smooth_top,) = viz.ax_top.plot([], [], "lime", lw=2.0, alpha=0.0)

    fly_trail = viz.create_trail_artists()
    viz.ax3d.legend(fontsize=7, loc="upper left")

    ax_d = viz.setup_data_axes(title="PRM Stats", ylabel="Count")
    ax_d.set_xlim(0, 3)
    n_nodes = len(planner.nodes)
    n_edges_total = len(edge_segs_3d)
    ax_d.set_ylim(0, max(n_nodes, n_edges_total) * 1.2)
    ax_d.bar(
        [0, 1, 2],
        [n_nodes, n_edges_total, len(raw_path)],
        color=["cyan", "teal", "blue"],
        alpha=0.6,
    )
    ax_d.set_xticks([0, 1, 2])
    ax_d.set_xticklabels(["Nodes", "Edges", "Path"], fontsize=6)

    title = viz.ax3d.set_title("PRM Roadmap")
    anim = SimAnimator("prm_3d", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    def update(f: int) -> None:
        p1_sample_end = N_SAMPLE
        p1_edge_end = p1_sample_end + N_EDGE
        p1_search_end = p1_edge_end + N_SEARCH
        p2_end = p1_search_end + N_SMOOTH

        if f < algo_frames + N_SMOOTH * SEARCH_REPEAT:
            phase_f = f // SEARCH_REPEAT
        else:
            phase_f = p2_end

        if phase_f < p1_sample_end:
            frac = min(1.0, (phase_f + 1) / N_SAMPLE)
            node_scat_3d.set_alpha(frac * 0.6)
            node_scat_top.set_alpha(frac * 0.4)
            n_show = max(1, int(frac * n_nodes))
            title.set_text(f"Sampling — {n_show}/{n_nodes} nodes")
        elif phase_f < p1_edge_end:
            node_scat_3d.set_alpha(0.5)
            node_scat_top.set_alpha(0.3)
            ef = phase_f - p1_sample_end
            frac = min(1.0, (ef + 1) / N_EDGE)
            if edge_col_3d:
                edge_col_3d.set_alpha(frac * 0.25)
            if edge_col_top:
                edge_col_top.set_alpha(frac * 0.2)
            n_show = max(1, int(frac * n_edges_total))
            title.set_text(f"Building Edges — {n_show}/{n_edges_total}")
        elif phase_f < p1_search_end:
            sf = phase_f - p1_edge_end
            if edge_col_3d:
                edge_col_3d.set_alpha(0.1)
            if edge_col_top:
                edge_col_top.set_alpha(0.08)
            if sf >= N_SEARCH // 2:
                raw_3d.set_alpha(1.0)
                raw_3d.set_data(raw_path[:, 0], raw_path[:, 1])
                raw_3d.set_3d_properties(raw_path[:, 2])
                raw_top.set_alpha(1.0)
                raw_top.set_data(raw_path[:, 0], raw_path[:, 1])
                title.set_text(f"Dijkstra — Path found ({len(raw_path)} nodes)")
            else:
                title.set_text("Dijkstra Search...")
        elif phase_f < p2_end:
            sf = phase_f - p1_search_end
            if sf >= N_SMOOTH // 2:
                smooth_3d.set_alpha(1.0)
                smooth_3d.set_data(smooth[:, 0], smooth[:, 1])
                smooth_3d.set_3d_properties(smooth[:, 2])
                smooth_top.set_alpha(1.0)
                smooth_top.set_data(smooth[:, 0], smooth[:, 1])
                raw_3d.set_alpha(0.2)
                raw_top.set_alpha(0.2)
                title.set_text("Smoothed Path (RDP + Resample)")
            else:
                title.set_text("Raw PRM Path")
        else:
            fi = f - (algo_frames + N_SMOOTH * SEARCH_REPEAT)
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Quadrotor Following PRM Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
