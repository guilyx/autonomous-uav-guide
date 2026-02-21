# Erwin Lejeune - 2026-02-15
"""Flight operations demo: full mission with state machine.

Demonstrates the complete mission lifecycle using the StateManager:
ARM -> TAKEOFF -> OFFBOARD (fly path via pure pursuit) -> LOITER -> LAND.

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
from uav_sim.simulations.common import CRUISE_ALT, WORLD_SIZE, frame_indices
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

START = np.array([4.0, 4.0, CRUISE_ALT])
GOAL = np.array([26.0, 26.0, CRUISE_ALT])


def main() -> None:
    _, buildings = default_world()

    mission_path = plan_through_obstacles(buildings, START, GOAL, world_size=int(WORLD_SIZE))
    if mission_path is None:
        mission_path = np.array([START, GOAL])

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
    fly_timeout = 120.0
    for _ in range(int(fly_timeout / dt)):
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

    speed = np.linalg.norm(states[:, 6:9], axis=1)
    dist_to_goal = np.linalg.norm(pos - GOAL, axis=1)

    idx = frame_indices(len(states))
    n_frames = len(idx)

    viz = ThreePanelViz(title="Flight Operations Demo — Full Mission", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(mission_path, color="blue", lw=1.0, alpha=0.4, label="A* Path")
    viz.mark_start_goal(START, GOAL)

    trail = viz.create_trail_artists()

    times = np.arange(len(states)) * dt
    ax_d = viz.setup_data_axes(title="Flight Data", ylabel="[m] / [m/s]")
    ax_d.set_xlim(0, times[-1])
    ax_d.set_ylim(0, max(1.0, max(speed.max(), dist_to_goal.max()) * 1.1))
    (l_dist,) = ax_d.plot([], [], "r-", lw=0.8, label="dist to goal")
    (l_spd,) = ax_d.plot([], [], "b-", lw=0.8, label="speed")
    ax_d.legend(fontsize=5, loc="upper right")

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
        phase = _current_phase(k)
        title.set_text(f"Flight Ops — {phase}")
        l_dist.set_data(times[:k], dist_to_goal[:k])
        l_spd.set_data(times[:k], speed[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
