# Erwin Lejeune - 2026-02-15
"""Pure Pursuit 3D path tracking: 3-panel demonstration.

Plans an obstacle-aware path with A*, then uses the StateManager to arm,
take off, and fly the path with adaptive pure pursuit.  The carrot
(look-ahead) point is visualised on all views.

Reference: R. C. Coulter, "Implementation of the Pure Pursuit Path
Tracking Algorithm," CMU-RI-TR-92-01, 1992.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.control import StateManager
from uav_sim.environment import default_world
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.simulations.common import CRUISE_ALT, WORLD_SIZE, frame_indices
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

START = np.array([3.0, 3.0, CRUISE_ALT])
GOAL = np.array([27.0, 27.0, CRUISE_ALT])


def main() -> None:
    _, buildings = default_world()

    waypoints = plan_through_obstacles(buildings, START, GOAL, world_size=int(WORLD_SIZE))
    if waypoints is None:
        waypoints = np.array([START, GOAL])

    quad = Quadrotor()
    quad.reset(position=np.array([START[0], START[1], 0.0]))

    sm = StateManager(quad)
    sm.arm()
    sm.run_takeoff(altitude=CRUISE_ALT, dt=0.005, timeout=10.0)

    pursuit = PurePursuit3D(lookahead=2.0, waypoint_threshold=1.5, adaptive=True, smoothing=0.3)
    sm.offboard()
    dt = 0.005
    for _ in range(int(120.0 / dt)):
        vel = quad.velocity
        target = pursuit.compute_target(quad.position, waypoints, velocity=vel)
        sm.set_position_target(target)
        sm.step(dt)
        if pursuit.is_path_complete(quad.position, waypoints):
            break

    sm.run_land(dt=dt, timeout=8.0)

    states = np.array(sm.states)
    pos = states[:, :3]
    times = np.arange(len(states)) * dt
    speed = np.linalg.norm(states[:, 6:9], axis=1)
    dist_to_goal = np.linalg.norm(pos - GOAL, axis=1)

    pursuit_viz = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    pursuit_viz.reset()
    carrot_points = np.zeros((len(states), 3))
    for i in range(len(states)):
        carrot_points[i] = pursuit_viz.compute_target(pos[i], waypoints, velocity=states[i, 6:9])
        if pursuit_viz.is_path_complete(pos[i], waypoints):
            carrot_points[i:] = waypoints[-1]
            break

    idx = frame_indices(len(states))
    n_frames = len(idx)

    viz = ThreePanelViz(title="Pure Pursuit 3D — Carrot Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(waypoints, color="blue", lw=1.0, alpha=0.4, label="A* Path")
    viz.mark_start_goal(START, GOAL)

    trail = viz.create_trail_artists()
    (carrot_3d,) = viz.ax3d.plot([], [], [], "m*", ms=12, zorder=6, label="Carrot")
    (carrot_top,) = viz.ax_top.plot([], [], "m*", ms=10, zorder=6)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    ax_d = viz.setup_data_axes(title="Flight Data", ylabel="[m] / [m/s]")
    ax_d.set_xlim(0, times[-1])
    ax_d.set_ylim(0, max(1.0, max(speed.max(), dist_to_goal.max()) * 1.1))
    (l_dist,) = ax_d.plot([], [], "r-", lw=0.8, label="dist to goal")
    (l_spd,) = ax_d.plot([], [], "b-", lw=0.8, label="speed")
    ax_d.legend(fontsize=5, loc="upper right")

    anim = SimAnimator("pure_pursuit", out_dir=Path(__file__).parent)
    anim._fig = viz.fig
    title = viz.ax3d.set_title("Pure Pursuit")

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        cp = carrot_points[k]
        carrot_3d.set_data([cp[0]], [cp[1]])
        carrot_3d.set_3d_properties([cp[2]])
        carrot_top.set_data([cp[0]], [cp[1]])
        l_dist.set_data(times[:k], dist_to_goal[:k])
        l_spd.set_data(times[:k], speed[:k])
        title.set_text(f"Pure Pursuit — t={times[k]:.1f}s  to_goal={dist_to_goal[k]:.1f}m")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
