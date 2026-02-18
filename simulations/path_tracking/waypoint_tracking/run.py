# Erwin Lejeune - 2026-02-15
"""Waypoint tracking with Pure Pursuit + PID: 3-panel live simulation.

Plans an obstacle-aware path via A*, then the drone takes off and
follows the full sequence of waypoints using pure pursuit for the
carrot point and cascaded PID for low-level control.

Reference: R. C. Coulter, "Implementation of the Pure Pursuit Path
Tracking Algorithm," CMU-RI-TR-92-01, 1992.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0
START = np.array([3.0, 3.0, CRUISE_ALT])
GOAL = np.array([27.0, 27.0, CRUISE_ALT])


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
    )
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=5, seed=9)

    waypoints = plan_through_obstacles(buildings, START, GOAL, world_size=int(WORLD_SIZE))
    if waypoints is None:
        print("No path found!")
        return

    quad = Quadrotor()
    quad.reset(position=np.array([START[0], START[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    states = fly_mission(
        quad,
        ctrl,
        waypoints,
        cruise_alt=CRUISE_ALT,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=3.0,
        landing_duration=3.0,
        loiter_duration=1.0,
    )
    pos = states[:, :3]

    # ── visualisation ──────────────────────────────────────────────────
    skip = max(1, len(states) // 200)
    idx = list(range(0, len(states), skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="Waypoint Tracking (Pure Pursuit + PID)", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(waypoints, color="blue", lw=1.0, alpha=0.5, label="A* Path")
    viz.mark_start_goal(START, GOAL)

    trail = viz.create_trail_artists()
    anim = SimAnimator("waypoint_tracking", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
