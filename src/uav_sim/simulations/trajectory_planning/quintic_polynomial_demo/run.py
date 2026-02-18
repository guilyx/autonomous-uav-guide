# Erwin Lejeune - 2026-02-15
"""Quintic polynomial planning: 3-panel, two-phase visualisation.

Phase 1 — Algorithm: quintic polynomial with full boundary conditions
           (position, velocity, acceleration) at start and goal. Shows
           the trajectory and its velocity / acceleration profiles.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit along trajectory -> land.

Reference: K. Takahashi, T. Scheuer, "Motion Planning in a Plane Using
Generalized Voronoi Diagrams," IEEE T-RA, 1989.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_planning.quintic_polynomial import QuinticPolynomialPlanner, QuinticState
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def main() -> None:
    world, buildings = default_world()

    start_state = QuinticState(
        pos=np.array([3.0, 3.0, CRUISE_ALT]),
        vel=np.array([1.5, 0.0, 0.0]),
        acc=np.zeros(3),
    )
    goal_state = QuinticState(
        pos=np.array([27.0, 27.0, CRUISE_ALT + 2]),
        vel=np.array([0.0, 1.0, 0.0]),
        acc=np.zeros(3),
    )
    T = 12.0

    planner = QuinticPolynomialPlanner()
    coeffs = planner.generate(start_state, goal_state, T)
    ts, traj_pts, traj_vel, traj_acc = planner.evaluate(coeffs, T, dt=0.05)

    # Phase 2: fly
    quad = Quadrotor()
    quad.reset(position=np.array([start_state.pos[0], start_state.pos[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    flight_states = fly_mission(
        quad,
        ctrl,
        traj_pts,
        cruise_alt=CRUISE_ALT,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.5,
        landing_duration=2.5,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    # ── Animation ─────────────────────────────────────────────────────
    n_traj = len(traj_pts)
    traj_step = max(1, n_traj // 80)
    traj_frames = list(range(0, n_traj, traj_step))
    fly_step = max(1, len(flight_pos) // 100)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_tf = len(traj_frames)
    n_ff = len(fly_frames)
    total = n_tf + n_ff

    viz = ThreePanelViz(title="Quintic Polynomial Trajectory", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.mark_start_goal(start_state.pos, goal_state.pos)

    (traj_3d,) = viz.ax3d.plot([], [], [], "b-", lw=2, alpha=0.7, label="Quintic")
    (traj_dot_3d,) = viz.ax3d.plot([], [], [], "bo", ms=5)
    (traj_top,) = viz.ax_top.plot([], [], "b-", lw=1.5, alpha=0.7)
    (traj_side,) = viz.ax_side.plot([], [], "b-", lw=1.5, alpha=0.7)

    # Velocity arrows in top-down every few steps
    arrow_every = max(1, n_traj // 12)

    fly_trail = viz.create_trail_artists()
    viz.ax3d.legend(fontsize=7, loc="upper left")
    title = viz.ax3d.set_title("Phase 1: Quintic Polynomial Generation")

    anim = SimAnimator("quintic_polynomial_demo", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    quiver_arts: list = []

    def update(f: int) -> None:
        for q in quiver_arts:
            q.remove()
        quiver_arts.clear()

        if f < n_tf:
            k = traj_frames[f]
            traj_3d.set_data(traj_pts[: k + 1, 0], traj_pts[: k + 1, 1])
            traj_3d.set_3d_properties(traj_pts[: k + 1, 2])
            traj_dot_3d.set_data([traj_pts[k, 0]], [traj_pts[k, 1]])
            traj_dot_3d.set_3d_properties([traj_pts[k, 2]])
            traj_top.set_data(traj_pts[: k + 1, 0], traj_pts[: k + 1, 1])
            traj_side.set_data(traj_pts[: k + 1, 0], traj_pts[: k + 1, 2])

            for i in range(0, k + 1, arrow_every):
                vn = np.linalg.norm(traj_vel[i])
                if vn > 0.01:
                    sc = min(2.0, 1.0)
                    q = viz.ax3d.quiver(
                        traj_pts[i, 0],
                        traj_pts[i, 1],
                        traj_pts[i, 2],
                        traj_vel[i, 0] * sc,
                        traj_vel[i, 1] * sc,
                        traj_vel[i, 2] * sc,
                        color="magenta",
                        linewidth=1,
                        arrow_length_ratio=0.3,
                        alpha=0.5,
                    )
                    quiver_arts.append(q)
            title.set_text(f"Phase 1: Quintic — t = {ts[k]:.1f}s / {T:.0f}s")
        else:
            traj_dot_3d.set_data([], [])
            traj_dot_3d.set_3d_properties([])
            fi = f - n_tf
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Phase 2: Quadrotor Flying Quintic Trajectory")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
