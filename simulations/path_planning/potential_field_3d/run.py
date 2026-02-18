# Erwin Lejeune - 2026-02-17
"""Potential Field 3D: two-phase visualisation.

Phase 1 — Algorithm: step-by-step gradient descent showing force vectors.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit along planned path -> land.

Reference: O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and
Mobile Robots," IJRR, 1986. DOI: 10.1177/027836498600500106
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_planning.potential_field_3d import PotentialField3D
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
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([8.0, 8.0, 5.0])
    obstacles = [
        (np.array([3.0, 3.0, 2.5]), 1.5),
        (np.array([5.0, 6.0, 3.0]), 1.2),
        (np.array([6.0, 2.0, 4.0]), 1.0),
    ]
    planner = PotentialField3D(step_size=0.3, max_iter=300)
    path = planner.plan(start, goal, obstacles)
    if path is None or len(path) < 2:
        print("Planning failed!")
        return
    path_pts = np.array(path)

    # Compute force vectors at each path point for visualisation
    forces = np.zeros_like(path_pts)
    for i, pt in enumerate(path_pts):
        f_att = planner._attractive_force(pt, goal)
        f_rep = planner._repulsive_force(pt, obstacles)
        forces[i] = f_att + f_rep

    # Smooth the potential-field path for flyable tracking
    flight_path = smooth_path_3d(path_pts, epsilon=0.3, min_spacing=0.3)

    # ── Phase 2: fly mission via pure pursuit ─────────────────────────────
    quad = Quadrotor()
    quad.reset(position=np.array([start[0], start[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=1.0, waypoint_threshold=0.4, adaptive=True)
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

    # ── Animation ─────────────────────────────────────────────────────────
    n_plan = len(path_pts)
    plan_step = max(1, n_plan // 100)
    plan_frames = list(range(0, n_plan, plan_step))
    fly_step = max(1, len(flight_pos) // 120)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_pf = len(plan_frames)
    n_ff = len(fly_frames)
    total = n_pf + n_ff

    anim = SimAnimator("potential_field_3d", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(10, 7))
    anim._fig = fig
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.scatter(*start, c="green", s=120, marker="^", label="Start", zorder=5)
    ax.scatter(*goal, c="red", s=120, marker="*", label="Goal", zorder=5)
    for oc, orad in obstacles:
        anim.draw_sphere(ax, oc, orad, color="red", alpha=0.2)
    anim.set_equal_3d(
        ax,
        np.vstack([path_pts, flight_pos]) if len(flight_pos) > 0 else path_pts,
        pad=2.0,
    )
    ax.legend(fontsize=7, loc="upper left")

    (plan_trail,) = ax.plot([], [], [], "c-", lw=2.0, alpha=0.7)
    (plan_dot,) = ax.plot([], [], [], "co", ms=6)
    quiver_artists: list = []
    (fly_trail,) = ax.plot([], [], [], "orange", lw=1.8)
    title = ax.set_title("Phase 1: Potential Field Descent")
    vehicle_arts: list = []

    def update(f):
        for q in quiver_artists:
            q.remove()
        quiver_artists.clear()
        clear_vehicle_artists(vehicle_arts)

        if f < n_pf:
            k = plan_frames[f]
            plan_trail.set_data(path_pts[: k + 1, 0], path_pts[: k + 1, 1])
            plan_trail.set_3d_properties(path_pts[: k + 1, 2])
            plan_dot.set_data([path_pts[k, 0]], [path_pts[k, 1]])
            plan_dot.set_3d_properties([path_pts[k, 2]])
            fv = forces[k]
            fn = np.linalg.norm(fv)
            if fn > 0.01:
                sc = min(1.0, 0.5 / fn)
                q = ax.quiver(
                    path_pts[k, 0],
                    path_pts[k, 1],
                    path_pts[k, 2],
                    fv[0] * sc,
                    fv[1] * sc,
                    fv[2] * sc,
                    color="magenta",
                    linewidth=2,
                    arrow_length_ratio=0.3,
                )
                quiver_artists.append(q)
            title.set_text(f"Phase 1: Potential Field — step {k + 1}/{n_plan}")
        else:
            plan_dot.set_data([], [])
            plan_dot.set_3d_properties([])
            fi = f - n_pf
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            fly_trail.set_data(flight_pos[:k, 0], flight_pos[:k, 1])
            fly_trail.set_3d_properties(flight_pos[:k, 2])
            R = Quadrotor.rotation_matrix(*flight_states[k, 3:6])
            vehicle_arts.extend(draw_quadrotor_3d(ax, flight_pos[k], R, size=0.5))
            title.set_text("Phase 2: Quadrotor Following Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
