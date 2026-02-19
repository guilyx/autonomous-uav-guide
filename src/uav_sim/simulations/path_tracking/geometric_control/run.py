# Erwin Lejeune - 2026-02-15
"""Geometric SO(3) controller: 3-panel attitude recovery + tracking.

Demonstrates the geometric controller recovering from an initial attitude
perturbation and then holding hover in a 30x30x30 urban world.

Reference: T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of
a Quadrotor UAV on SE(3)," CDC, 2010. DOI: 10.1109/CDC.2010.5717652
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.geometric_controller import GeometricController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
TARGET = np.array([15.0, 15.0, 12.0])


def main() -> None:
    world, buildings = default_world()

    quad = Quadrotor()
    quad.reset(position=TARGET.copy(), euler=np.array([0.12, -0.10, 0.0]))
    hover_f = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hover_f))

    ctrl = GeometricController()

    dt, dur = 0.005, 12.0
    steps = int(dur / dt)
    states = np.zeros((steps, 12))
    times = np.zeros(steps)
    for i in range(steps):
        times[i] = i * dt
        states[i] = quad.state
        quad.step(ctrl.compute(quad.state, TARGET), dt)

    # ── visualisation ──────────────────────────────────────────────────
    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="Geometric SO(3) — Attitude Recovery", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.ax3d.scatter(*TARGET, c="lime", s=100, marker="*", zorder=5, label="Target")
    viz.ax_top.plot(TARGET[0], TARGET[1], "g*", ms=12, zorder=5)
    viz.ax_side.plot(TARGET[0], TARGET[2], "g*", ms=12, zorder=5)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    trail = viz.create_trail_artists()
    pos = states[:, :3]

    anim = SimAnimator("geometric_control", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
