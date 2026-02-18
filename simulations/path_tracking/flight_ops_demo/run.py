# Erwin Lejeune - 2026-02-15
"""Flight operations demo: 3-panel visualisation of a full mission.

Plans an obstacle-aware path via A*, then demonstrates the complete
flight-ops pipeline: takeoff -> fly path -> loiter -> land.

Reference: Generic multirotor operation sequence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
from uav_sim.path_tracking.flight_ops import (
    fly_path,
    landing,
    loiter,
    takeoff,
)
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 14.0
START = np.array([4.0, 4.0, CRUISE_ALT])
GOAL = np.array([26.0, 26.0, CRUISE_ALT])


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
    )
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=5, seed=55)

    mission_path = plan_through_obstacles(buildings, START, GOAL, world_size=int(WORLD_SIZE))
    if mission_path is None:
        print("No path found!")
        return

    loiter_pos = mission_path[-1].copy()

    quad = Quadrotor()
    quad.reset(position=np.array([START[0], START[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    states: list[np.ndarray] = []
    phase_ends: list[tuple[int, str]] = []

    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=0.005, duration=3.0, states=states)
    phase_ends.append((len(states), "Takeoff"))

    fly_path(quad, ctrl, mission_path, dt=0.005, pursuit=pursuit, timeout=60.0, states=states)
    phase_ends.append((len(states), "Fly Path"))

    loiter(quad, ctrl, loiter_pos, dt=0.005, duration=3.0, states=states)
    phase_ends.append((len(states), "Loiter"))

    landing(quad, ctrl, dt=0.005, duration=4.0, states=states)
    phase_ends.append((len(states), "Landing"))

    all_states = np.array(states) if states else np.zeros((1, 12))
    pos = all_states[:, :3]

    # ── visualisation ──────────────────────────────────────────────────
    skip = max(1, len(all_states) // 200)
    idx = list(range(0, len(all_states), skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="Flight Operations Demo — Full Mission", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(mission_path, color="blue", lw=1.0, alpha=0.4, label="A* Path")
    viz.mark_start_goal(START, GOAL)

    trail = viz.create_trail_artists()
    title = viz.ax3d.set_title("Takeoff")

    anim = SimAnimator("flight_ops_demo", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def _current_phase(k: int) -> str:
        for end, name in phase_ends:
            if k < end:
                return name
        return phase_ends[-1][1] if phase_ends else ""

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], all_states[k, 3:6], size=1.5)
        title.set_text(f"Flight Ops — {_current_phase(k)}")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
