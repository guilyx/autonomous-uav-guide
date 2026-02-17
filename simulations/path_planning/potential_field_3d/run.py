# Erwin Lejeune - 2026-02-17
"""Potential Field 3D: two-phase visualisation.

Phase 1 — Algorithm: step-by-step gradient descent showing force vectors.
Phase 2 — Platform: quadrotor following the planned path with full dynamics.

Reference: O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and
Mobile Robots," IJRR, 1986. DOI: 10.1177/027836498600500106
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_planning.potential_field_3d import PotentialField3D
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

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

    # Phase 2: quadrotor following path
    quad = Quadrotor()
    quad.reset(position=path_pts[0])
    ctrl = CascadedPIDController()
    dt = 0.005
    flight_pos_list: list[np.ndarray] = []
    flight_euler_list: list[np.ndarray] = []
    for wp in path_pts[::2]:
        for _ in range(60):
            s = quad.state
            p = s[:3].copy()
            if np.any(np.isnan(p)) or np.any(np.abs(p) > 100):
                break
            flight_pos_list.append(p)
            flight_euler_list.append(s[3:6].copy())
            quad.step(ctrl.compute(s, wp, dt=dt), dt)
    flight_pos = np.array(flight_pos_list) if flight_pos_list else path_pts[:1]
    flight_euler = np.array(flight_euler_list) if flight_euler_list else np.zeros((1, 3))

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
        ax, np.vstack([path_pts, flight_pos]) if len(flight_pos) > 0 else path_pts, pad=2.0
    )
    ax.legend(fontsize=7, loc="upper left")

    (plan_trail,) = ax.plot([], [], [], "c-", lw=2.0, alpha=0.7)
    (plan_dot,) = ax.plot([], [], [], "co", ms=6)
    quiver_artists = []
    (fly_trail,) = ax.plot([], [], [], "orange", lw=1.8)
    (fly_dot,) = ax.plot([], [], [], "ko", ms=7)
    title = ax.set_title("Phase 1: Potential Field Descent")
    vehicle_arts: list = []

    def update(f):
        nonlocal quiver_artists
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
                scale = min(1.0, 0.5 / fn)
                q = ax.quiver(
                    path_pts[k, 0],
                    path_pts[k, 1],
                    path_pts[k, 2],
                    fv[0] * scale,
                    fv[1] * scale,
                    fv[2] * scale,
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
            fly_dot.set_data([flight_pos[k, 0]], [flight_pos[k, 1]])
            fly_dot.set_3d_properties([flight_pos[k, 2]])
            R = Quadrotor.rotation_matrix(*flight_euler[k])
            vehicle_arts.extend(draw_quadrotor_3d(ax, flight_pos[k], R, scale=30.0))
            title.set_text("Phase 2: Quadrotor Following Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
