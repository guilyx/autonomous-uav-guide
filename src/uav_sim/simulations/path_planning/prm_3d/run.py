# Erwin Lejeune - 2026-02-15
"""PRM 3D: 3-panel, two-phase visualisation.

Phase 1 — Algorithm: roadmap construction with progressive edge reveal,
           then Dijkstra query with path highlight.
           Path smoothing is shown as a distinct step.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit along smoothed path -> land.

Reference: L. E. Kavraki et al., "Probabilistic Roadmaps for Path Planning
in High-Dimensional Configuration Spaces," IEEE T-RA, 1996.
DOI: 10.1109/70.508439
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
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


def _box_to_sphere(b):
    centre = (b.min_corner + b.max_corner) / 2
    radius = float(np.linalg.norm(b.max_corner - b.min_corner)) / 2
    return (centre, radius)


def main() -> None:
    world = World(bounds_min=np.zeros(3), bounds_max=np.full(3, WORLD_SIZE))
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=5, seed=19)
    sphere_obs = [_box_to_sphere(b) for b in buildings]

    start = np.array([2.0, 2.0, 12.0])
    goal = np.array([28.0, 28.0, 12.0])

    planner = PRM3D(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
        obstacles=sphere_obs,
        n_samples=300,
        k_neighbours=12,
    )
    planner.build(seed=42)
    path = planner.plan(start, goal)
    if path is None:
        print("No path found!")
        return

    raw_path = np.array(path)
    smooth = smooth_path_3d(raw_path, epsilon=2.0, min_spacing=1.5)

    # Phase 2: fly
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

    # ── Animation ─────────────────────────────────────────────────────
    roadmap_pause = 15
    smooth_pause = 10
    fly_step = max(1, len(flight_pos) // 60)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_ff = len(fly_frames)
    total = roadmap_pause + smooth_pause + n_ff

    viz = ThreePanelViz(title="PRM 3D — Roadmap → Smooth → Flight", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.mark_start_goal(start, goal)

    # Draw roadmap edges
    for i, adj in enumerate(planner.edges):
        for j, _ in adj:
            if j > i:
                p1, p2 = planner.nodes[i], planner.nodes[j]
                viz.ax3d.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    "c-",
                    lw=0.2,
                    alpha=0.0,
                )
    edge_lines = viz.ax3d.lines[-len([e for adj in planner.edges for e in adj if e[0] > 0]) // 2 :]

    viz.ax3d.scatter(
        planner.nodes[:, 0],
        planner.nodes[:, 1],
        planner.nodes[:, 2],
        c="cyan",
        s=2,
        alpha=0.0,
        zorder=3,
    )
    node_scat = viz.ax3d.collections[-1]

    (raw_line_3d,) = viz.ax3d.plot([], [], [], "b-", lw=1.5, alpha=0.0)
    (smooth_line_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.5, alpha=0.0, label="Smoothed")
    (raw_line_top,) = viz.ax_top.plot([], [], "b-", lw=1.0, alpha=0.0)
    (smooth_line_top,) = viz.ax_top.plot([], [], "lime", lw=2.0, alpha=0.0)
    (raw_line_side,) = viz.ax_side.plot([], [], "b-", lw=1.0, alpha=0.0)
    (smooth_line_side,) = viz.ax_side.plot([], [], "lime", lw=2.0, alpha=0.0)

    fly_trail = viz.create_trail_artists()
    title = viz.ax3d.set_title("Phase 1: PRM Roadmap")

    anim = SimAnimator("prm_3d", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    def update(f: int) -> None:
        if f < roadmap_pause:
            frac = min(1.0, (f + 1) / (roadmap_pause * 0.6))
            node_scat.set_alpha(frac * 0.5)
            for ln in edge_lines:
                ln.set_alpha(frac * 0.15)
            if f >= roadmap_pause - 3:
                raw_line_3d.set_alpha(1.0)
                raw_line_3d.set_data(raw_path[:, 0], raw_path[:, 1])
                raw_line_3d.set_3d_properties(raw_path[:, 2])
                raw_line_top.set_alpha(1.0)
                raw_line_top.set_data(raw_path[:, 0], raw_path[:, 1])
                raw_line_side.set_alpha(1.0)
                raw_line_side.set_data(raw_path[:, 0], raw_path[:, 2])
            title.set_text(f"Phase 1: PRM Roadmap — {int(100 * (f + 1) / roadmap_pause)}%")
        elif f < roadmap_pause + smooth_pause:
            sf = f - roadmap_pause
            if sf < smooth_pause // 2:
                title.set_text("Raw PRM Path (Dijkstra)")
            else:
                smooth_line_3d.set_alpha(1.0)
                smooth_line_3d.set_data(smooth[:, 0], smooth[:, 1])
                smooth_line_3d.set_3d_properties(smooth[:, 2])
                smooth_line_top.set_alpha(1.0)
                smooth_line_top.set_data(smooth[:, 0], smooth[:, 1])
                smooth_line_side.set_alpha(1.0)
                smooth_line_side.set_data(smooth[:, 0], smooth[:, 2])
                raw_line_3d.set_alpha(0.2)
                raw_line_top.set_alpha(0.2)
                raw_line_side.set_alpha(0.2)
                title.set_text("Smoothed Path (RDP + Resample)")
        else:
            fi = f - roadmap_pause - smooth_pause
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Phase 2: Quadrotor Following PRM Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
