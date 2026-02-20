# Erwin Lejeune - 2026-02-15
"""Potential Field 3D: 3-panel, two-phase visualisation.

Phase 1 — Algorithm: step-by-step gradient descent with force vectors.
           Path smoothing is shown as an intermediate step.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit along path -> land.

Reference: O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and
Mobile Robots," IJRR, 1986. DOI: 10.1177/027836498600500106
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_planning.potential_field_3d import PotentialField3D
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
    world, buildings = default_world()
    sphere_obs = [_box_to_sphere(b) for b in buildings]

    start = np.array([2.0, 2.0, 12.0])
    goal = np.array([28.0, 28.0, 12.0])

    planner = PotentialField3D(
        zeta=1.0, eta=200.0, rho0=5.0, step_size=0.5, max_iter=500, goal_tol=1.0
    )
    path = planner.plan(start, goal, sphere_obs)
    if path is None or len(path) < 2:
        print("Planning failed!")
        return

    raw_path = np.array(path)
    forces = np.zeros_like(raw_path)
    for i, pt in enumerate(raw_path):
        f_att = planner._attractive_force(pt, goal)
        f_rep = planner._repulsive_force(pt, sphere_obs)
        forces[i] = f_att + f_rep

    smooth_path = smooth_path_3d(raw_path, epsilon=1.5, min_spacing=1.0)

    # Phase 2: fly smoothed path
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

    # ── Animation ─────────────────────────────────────────────────────
    n_plan = len(raw_path)
    plan_step = max(1, n_plan // 80)
    plan_frames = list(range(0, n_plan, plan_step))
    smooth_pause = 15
    fly_step = max(1, len(flight_pos) // 100)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_pf = len(plan_frames)
    n_ff = len(fly_frames)
    total = n_pf + smooth_pause + n_ff

    viz = ThreePanelViz(
        title="Potential Field 3D — Descent → Smooth → Flight", world_size=WORLD_SIZE
    )
    viz.draw_buildings(buildings)
    viz.mark_start_goal(start, goal)

    (plan_trail_3d,) = viz.ax3d.plot([], [], [], "c-", lw=2.0, alpha=0.7)
    (plan_dot_3d,) = viz.ax3d.plot([], [], [], "co", ms=6)
    (plan_trail_top,) = viz.ax_top.plot([], [], "c-", lw=1.5, alpha=0.7)
    (plan_trail_side,) = viz.ax_side.plot([], [], "c-", lw=1.5, alpha=0.7)

    (smooth_line_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.5, alpha=0.0, label="Smoothed")
    (smooth_line_top,) = viz.ax_top.plot([], [], "lime", lw=2.0, alpha=0.0)
    (smooth_line_side,) = viz.ax_side.plot([], [], "lime", lw=2.0, alpha=0.0)

    quiver_artists: list = []
    fly_trail = viz.create_trail_artists()
    title = viz.ax3d.set_title("Phase 1: Potential Field Descent")

    anim = SimAnimator("potential_field_3d", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        for q in quiver_artists:
            q.remove()
        quiver_artists.clear()

        if f < n_pf:
            k = plan_frames[f]
            plan_trail_3d.set_data(raw_path[: k + 1, 0], raw_path[: k + 1, 1])
            plan_trail_3d.set_3d_properties(raw_path[: k + 1, 2])
            plan_dot_3d.set_data([raw_path[k, 0]], [raw_path[k, 1]])
            plan_dot_3d.set_3d_properties([raw_path[k, 2]])
            plan_trail_top.set_data(raw_path[: k + 1, 0], raw_path[: k + 1, 1])
            plan_trail_side.set_data(raw_path[: k + 1, 0], raw_path[: k + 1, 2])
            fv = forces[k]
            fn = np.linalg.norm(fv)
            if fn > 0.01:
                sc = min(2.0, 1.0 / fn)
                q = viz.ax3d.quiver(
                    raw_path[k, 0],
                    raw_path[k, 1],
                    raw_path[k, 2],
                    fv[0] * sc,
                    fv[1] * sc,
                    fv[2] * sc,
                    color="magenta",
                    linewidth=2,
                    arrow_length_ratio=0.3,
                )
                quiver_artists.append(q)
            title.set_text(f"Phase 1: Potential Field — step {k + 1}/{n_plan}")
        elif f < n_pf + smooth_pause:
            plan_dot_3d.set_data([], [])
            plan_dot_3d.set_3d_properties([])
            sf = f - n_pf
            if sf < smooth_pause // 2:
                title.set_text("Raw Potential Field Path")
            else:
                smooth_line_3d.set_alpha(1.0)
                smooth_line_3d.set_data(smooth_path[:, 0], smooth_path[:, 1])
                smooth_line_3d.set_3d_properties(smooth_path[:, 2])
                smooth_line_top.set_alpha(1.0)
                smooth_line_top.set_data(smooth_path[:, 0], smooth_path[:, 1])
                smooth_line_side.set_alpha(1.0)
                smooth_line_side.set_data(smooth_path[:, 0], smooth_path[:, 2])
                plan_trail_3d.set_alpha(0.2)
                plan_trail_top.set_alpha(0.2)
                plan_trail_side.set_alpha(0.2)
                title.set_text("Smoothed Path (RDP + Resample)")
        else:
            plan_dot_3d.set_data([], [])
            plan_dot_3d.set_3d_properties([])
            fi = f - n_pf - smooth_pause
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Phase 2: Quadrotor Following Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
