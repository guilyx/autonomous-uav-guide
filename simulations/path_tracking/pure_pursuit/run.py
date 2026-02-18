# Erwin Lejeune - 2026-02-15
"""Pure Pursuit 3D path tracking: 3-panel demonstration.

Plans an obstacle-aware path with A*, then demonstrates the adaptive
pure-pursuit controller following it through an urban environment.
The carrot (look-ahead) point is visualised on all three views.

Reference: R. C. Coulter, "Implementation of the Pure Pursuit Path
Tracking Algorithm," CMU-RI-TR-92-01, 1992.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
from uav_sim.path_tracking.flight_ops import fly_path, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0
START = np.array([3.0, 3.0, CRUISE_ALT])
GOAL = np.array([27.0, 27.0, CRUISE_ALT])


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
    )
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=5, seed=17)

    waypoints = plan_through_obstacles(buildings, START, GOAL, world_size=int(WORLD_SIZE))
    if waypoints is None:
        print("No path found!")
        return

    quad = Quadrotor()
    quad.reset(position=np.array([START[0], START[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    states: list[np.ndarray] = []
    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=0.005, duration=3.0, states=states)
    fly_path(quad, ctrl, waypoints, dt=0.005, pursuit=pursuit, timeout=60.0, states=states)

    all_states = np.array(states) if states else np.zeros((1, 12))
    pos = all_states[:, :3]

    # Recompute carrot points for visualisation
    pursuit_viz = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    pursuit_viz.reset()
    carrot_points = np.zeros((len(all_states), 3))
    for i in range(len(all_states)):
        vel = all_states[i, 6:9]
        carrot_points[i] = pursuit_viz.compute_target(pos[i], waypoints, velocity=vel)
        if pursuit_viz.is_path_complete(pos[i], waypoints):
            carrot_points[i:] = waypoints[-1]
            break

    # ── visualisation ──────────────────────────────────────────────────
    skip = max(1, len(all_states) // 200)
    idx = list(range(0, len(all_states), skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="Pure Pursuit 3D — Carrot Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(waypoints, color="blue", lw=1.0, alpha=0.4, label="A* Path")
    viz.mark_start_goal(START, GOAL)

    trail = viz.create_trail_artists()
    (carrot_3d,) = viz.ax3d.plot([], [], [], "m*", ms=12, zorder=6, label="Carrot")
    (carrot_top,) = viz.ax_top.plot([], [], "m*", ms=10, zorder=6)
    (carrot_side,) = viz.ax_side.plot([], [], "m*", ms=10, zorder=6)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    anim = SimAnimator("pure_pursuit", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], all_states[k, 3:6], size=1.5)
        cp = carrot_points[k]
        carrot_3d.set_data([cp[0]], [cp[1]])
        carrot_3d.set_3d_properties([cp[2]])
        carrot_top.set_data([cp[0]], [cp[1]])
        carrot_side.set_data([cp[0]], [cp[2]])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
