# Erwin Lejeune - 2026-02-15
"""Waypoint tracking with Pure Pursuit + PID: 3-panel live simulation.

The drone takes off, then follows a sequence of waypoints using pure
pursuit for the carrot point and cascaded PID for low-level control.
This is the recommended pattern for all path-tracking simulations.

Reference: R. C. Coulter, "Implementation of the Pure Pursuit Path
Tracking Algorithm," CMU-RI-TR-92-01, 1992.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
    )
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=5, seed=9)

    waypoints = np.array(
        [
            [3.0, 3.0, CRUISE_ALT],
            [10.0, 3.0, CRUISE_ALT],
            [10.0, 15.0, CRUISE_ALT + 3],
            [20.0, 15.0, CRUISE_ALT],
            [20.0, 25.0, CRUISE_ALT + 2],
            [27.0, 27.0, CRUISE_ALT],
        ]
    )

    quad = Quadrotor()
    quad.reset(position=np.array([3.0, 3.0, 0.0]))
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
    viz.draw_path(waypoints, color="red", lw=1.0, alpha=0.5, label="Waypoints")

    # waypoint markers
    viz.ax3d.scatter(
        waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="red", s=60, marker="D", zorder=5
    )
    for i, wp in enumerate(waypoints):
        viz.ax3d.text(wp[0], wp[1], wp[2] + 1.0, f"WP{i}", fontsize=7, ha="center")
        viz.ax_top.plot(wp[0], wp[1], "rD", ms=5)
        viz.ax_side.plot(wp[0], wp[2], "rD", ms=5)

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
