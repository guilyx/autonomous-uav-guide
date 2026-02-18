# Erwin Lejeune - 2026-02-15
"""Minimum-snap trajectory: 3-panel, two-phase visualisation.

Phase 1 — Algorithm: A* plans obstacle-aware waypoints, then min-snap
           generates smooth trajectory segments through them.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit trajectory -> land.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011. DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import rdp_simplify
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_planning.min_snap import MinSnapTrajectory
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0
START = np.array([3.0, 3.0, CRUISE_ALT])
GOAL = np.array([27.0, 27.0, CRUISE_ALT])


def main() -> None:
    world, buildings = default_world()

    planned = plan_through_obstacles(buildings, START, GOAL, world_size=int(WORLD_SIZE))
    if planned is None:
        print("No path found!")
        return

    # Reduce waypoints for min-snap (it needs a small number of knot points)
    wps = rdp_simplify(planned, epsilon=3.0)
    if len(wps) < 2:
        wps = planned[:: max(1, len(planned) // 5)]
    # Ensure start and goal are exact
    wps[0] = START.copy()
    wps[-1] = GOAL.copy()

    seg_times = np.full(len(wps) - 1, 3.0)

    ms = MinSnapTrajectory()
    coeffs = ms.generate(wps, seg_times)
    _, traj_pts = ms.evaluate(coeffs, seg_times, dt=0.05)

    # Phase 2: fly
    quad = Quadrotor()
    quad.reset(position=np.array([START[0], START[1], 0.0]))
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

    viz = ThreePanelViz(title="Minimum-Snap Trajectory", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.mark_start_goal(START, GOAL)

    viz.ax3d.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c="red", s=80, marker="D", zorder=5)
    for i, wp in enumerate(wps):
        viz.ax3d.text(wp[0], wp[1], wp[2] + 1.0, f"WP{i}", fontsize=7, ha="center")
        viz.ax_top.plot(wp[0], wp[1], "rD", ms=5)
        viz.ax_side.plot(wp[0], wp[2], "rD", ms=5)

    (traj_3d,) = viz.ax3d.plot([], [], [], "b-", lw=2, alpha=0.7, label="Min-Snap")
    (traj_dot_3d,) = viz.ax3d.plot([], [], [], "bo", ms=5)
    (traj_top,) = viz.ax_top.plot([], [], "b-", lw=1.5, alpha=0.7)
    (traj_side,) = viz.ax_side.plot([], [], "b-", lw=1.5, alpha=0.7)

    fly_trail = viz.create_trail_artists()
    viz.ax3d.legend(fontsize=7, loc="upper left")
    title = viz.ax3d.set_title("Phase 1: Trajectory Generation")

    anim = SimAnimator("min_snap", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        if f < n_tf:
            k = traj_frames[f]
            traj_3d.set_data(traj_pts[: k + 1, 0], traj_pts[: k + 1, 1])
            traj_3d.set_3d_properties(traj_pts[: k + 1, 2])
            traj_dot_3d.set_data([traj_pts[k, 0]], [traj_pts[k, 1]])
            traj_dot_3d.set_3d_properties([traj_pts[k, 2]])
            traj_top.set_data(traj_pts[: k + 1, 0], traj_pts[: k + 1, 1])
            traj_side.set_data(traj_pts[: k + 1, 0], traj_pts[: k + 1, 2])
            seg = min(int(k / (n_traj / len(seg_times))), len(seg_times) - 1)
            title.set_text(f"Phase 1: Min-Snap — segment {seg + 1}/{len(seg_times)}")
        else:
            traj_dot_3d.set_data([], [])
            traj_dot_3d.set_3d_properties([])
            fi = f - n_tf
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Phase 2: Quadrotor Flying Min-Snap Trajectory")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
