# Erwin Lejeune - 2026-02-15
"""Flight operations demo: full mission with state machine.

Demonstrates the complete mission lifecycle using the StateManager:
ARM → TAKEOFF → OFFBOARD (fly path via pure pursuit) → LOITER → LAND.

Reference: Generic multirotor operation sequence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.control import StateManager
from uav_sim.environment import default_world
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
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
    _, buildings = default_world()

    mission_path = plan_through_obstacles(buildings, START, GOAL, world_size=int(WORLD_SIZE))
    if mission_path is None:
        print("No path found!")
        return

    quad = Quadrotor()
    quad.reset(position=np.array([START[0], START[1], 0.0]))

    sm = StateManager(quad)
    dt = 0.005
    phase_ends: list[tuple[int, str]] = []

    sm.arm()
    sm.run_takeoff(altitude=CRUISE_ALT, dt=dt, timeout=10.0)
    phase_ends.append((len(sm.states), "Takeoff"))

    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    sm.offboard()
    for _ in range(int(60.0 / dt)):
        target = pursuit.compute_target(quad.position, mission_path, velocity=quad.velocity)
        sm.set_position_target(target)
        sm.step(dt)
        if pursuit.is_path_complete(quad.position, mission_path):
            break
    phase_ends.append((len(sm.states), "Fly Path"))

    sm.loiter()
    for _ in range(int(3.0 / dt)):
        sm.step(dt)
    phase_ends.append((len(sm.states), "Loiter"))

    sm.run_land(dt=dt, timeout=10.0)
    phase_ends.append((len(sm.states), "Landing"))

    states = np.array(sm.states)
    pos = states[:, :3]

    skip = max(1, len(states) // 200)
    idx = list(range(0, len(states), skip))
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
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        title.set_text(f"Flight Ops — {_current_phase(k)}")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
