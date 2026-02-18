# Erwin Lejeune - 2026-02-18
"""Fixed-wing level flight and gentle banked turn demonstration.

Three panels (3D, top-down, side) showing the fixed-wing UAV performing
straight flight followed by a banked turn manoeuvre in a 30 m environment.

Reference: R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and
Practice," Princeton University Press, 2012.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.vehicles.fixed_wing import FixedWing
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_fixed_wing_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0


def main() -> None:
    fw = FixedWing()
    state = np.zeros(12)
    state[:3] = [3.0, 15.0, 15.0]
    state[6] = 8.0
    fw.reset(state=state)

    dt, duration = 0.002, 15.0
    steps = int(duration / dt)
    positions = np.zeros((steps, 3))
    eulers = np.zeros((steps, 3))

    for i in range(steps):
        positions[i] = fw.state[:3]
        eulers[i] = fw.state[3:6]
        t = i * dt
        throttle = 0.5
        if t < 4:
            elevator, aileron, rudder = -0.01, 0.0, 0.0
        elif t < 9:
            elevator, aileron, rudder = -0.01, 0.05, 0.01
        else:
            elevator, aileron, rudder = -0.01, -0.03, -0.005

        fw.step(np.array([elevator, aileron, rudder, throttle]), dt)

        if np.any(np.isnan(fw.state)):
            positions = positions[:i]
            eulers = eulers[:i]
            break

        pos = fw.state[:3]
        if np.any(pos < -5) or np.any(pos > WORLD_SIZE + 5):
            positions = positions[: i + 1]
            eulers = eulers[: i + 1]
            break

    n_steps = len(positions)
    if n_steps < 2:
        print("Fixed-wing simulation too short — skipping")
        return

    viz = ThreePanelViz(title="Fixed-Wing Flight", world_size=WORLD_SIZE)
    anim = SimAnimator("fixed_wing_flight", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="royalblue")

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    vehicle_arts_3d: list = []

    def update(f: int) -> None:
        k = idx[min(f, len(idx) - 1)]
        viz.update_trail(trail_arts, positions, k)

        clear_vehicle_artists(vehicle_arts_3d)
        R = Quadrotor.rotation_matrix(*eulers[k])
        vehicle_arts_3d.extend(draw_fixed_wing_3d(viz.ax3d, positions[k], R, scale=3.0))

        clear_vehicle_artists(viz._vehicle_arts_top)
        (dot_top,) = viz.ax_top.plot(positions[k, 0], positions[k, 1], "ko", ms=5, zorder=5)
        viz._vehicle_arts_top.append(dot_top)
        clear_vehicle_artists(viz._vehicle_arts_side)
        (dot_side,) = viz.ax_side.plot(positions[k, 0], positions[k, 2], "ko", ms=5, zorder=5)
        viz._vehicle_arts_side.append(dot_side)

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
