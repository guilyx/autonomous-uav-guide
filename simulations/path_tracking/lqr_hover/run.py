# Erwin Lejeune - 2026-02-15
"""LQR hover stabilisation: 3-panel live simulation.

Shows a quadrotor recovering from an offset position to a hover target
using a continuous-time LQR controller, rendered across 3D, top-down,
and side views with urban buildings.

Reference: B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear Quadratic
Methods," Prentice-Hall, 1990.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_tracking.flight_ops import init_hover
from uav_sim.path_tracking.lqr_controller import LQRController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
    )
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, seed=11)

    start = np.array([8.0, 6.0, 5.0])
    target_state = np.zeros(12)
    target_state[:3] = [15.0, 15.0, 12.0]

    quad = Quadrotor()
    quad.reset(position=start)
    init_hover(quad)
    ctrl = LQRController(
        mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
    )

    dt, duration = 0.005, 10.0
    steps = int(duration / dt)
    states = np.zeros((steps, 12))
    controls = np.zeros((steps, 4))
    times = np.zeros(steps)
    for i in range(steps):
        states[i] = quad.state
        times[i] = i * dt
        u = ctrl.compute(quad.state, target_state)
        controls[i] = u
        quad.step(u, dt)

    # ── visualisation ──────────────────────────────────────────────────
    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="LQR Hover Stabilisation", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.mark_start_goal(start, target_state[:3])

    trail = viz.create_trail_artists()
    pos = states[:, :3]

    anim = SimAnimator("lqr_hover", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
