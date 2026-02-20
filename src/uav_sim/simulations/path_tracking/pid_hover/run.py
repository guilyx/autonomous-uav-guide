# Erwin Lejeune - 2026-02-15
"""PID-controlled hover: 3-panel live simulation.

Three views (3D, top-down XY, side XZ) showing a quadrotor taking off
and hovering at a target point inside an urban environment using the
new layered control stack and state machine.

Reference: L. R. G. Carrillo et al., "Quad Rotorcraft Control," Springer, 2013.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.control import StateManager
from uav_sim.environment import default_world
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
TARGET = np.array([15.0, 15.0, 12.0])


def main() -> None:
    _, buildings = default_world()

    quad = Quadrotor()
    quad.reset(position=np.array([2.0, 2.0, 0.0]))

    sm = StateManager(quad)
    sm.arm()
    sm.run_takeoff(altitude=TARGET[2], dt=0.005, timeout=10.0)
    sm.fly_to(TARGET, dt=0.005, threshold=0.5, timeout=15.0)

    for _ in range(int(3.0 / 0.005)):
        sm.step(0.005)

    states = np.array(sm.states)
    n_total = len(states)

    skip = max(1, n_total // 200)
    idx = list(range(0, n_total, skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="PID Hover — Live Simulation", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.ax3d.scatter(*TARGET, c="lime", s=100, marker="*", zorder=5, label="Target")
    viz.ax_top.plot(TARGET[0], TARGET[1], "g*", ms=12, zorder=5)
    viz.ax_side.plot(TARGET[0], TARGET[2], "g*", ms=12, zorder=5)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    trail = viz.create_trail_artists()
    pos = states[:, :3]
    anim = SimAnimator("pid_hover", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
