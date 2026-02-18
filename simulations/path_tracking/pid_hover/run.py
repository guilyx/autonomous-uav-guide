# Erwin Lejeune - 2026-02-15
"""PID-controlled hover: 3-panel live simulation.

Three views (3D, top-down XY, side XZ) showing a quadrotor climbing
from ground level to a target hover point inside an urban environment.

Reference: L. R. G. Carrillo et al., "Quad Rotorcraft Control," Springer, 2013.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_tracking.flight_ops import _limit_target
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
TARGET = np.array([15.0, 15.0, 12.0])


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
    )
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, seed=7)

    quad = Quadrotor()
    quad.reset(position=np.array([2.0, 2.0, 0.0]))
    ctrl = CascadedPIDController()
    dt, duration = 0.005, 8.0
    steps = int(duration / dt)
    states = np.zeros((steps, 12))
    times = np.zeros(steps)
    for i in range(steps):
        states[i] = quad.state
        times[i] = i * dt
        cmd = _limit_target(quad.state[:3], TARGET)
        quad.step(ctrl.compute(quad.state, cmd, dt=dt), dt)

    # ── visualisation ──────────────────────────────────────────────────
    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
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
