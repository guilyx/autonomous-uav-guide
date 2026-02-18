# Erwin Lejeune - 2026-02-18
"""VTOL tilt-rotor transition: hover -> cruise -> hover.

Three panels (3D, top-down, side) in a 30 m environment showing the VTOL
performing a tilt transition from hover to forward flight and back.

Reference: R. Bapst et al., "Design and Implementation of an Unmanned
Tail-Sitter," IROS, 2015.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.vehicles.vtol import Tiltrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0


def main() -> None:
    vtol = Tiltrotor()
    state = np.zeros(12)
    state[:3] = [5.0, 15.0, 15.0]
    vtol.reset(state=state)

    dt, duration = 0.005, 20.0
    steps = int(duration / dt)
    positions = np.zeros((steps, 3))
    eulers = np.zeros((steps, 3))
    tilt_angles = np.zeros(steps)

    g = vtol.vtol_params.gravity
    m = vtol.vtol_params.mass

    for i in range(steps):
        positions[i] = vtol.state[:3]
        eulers[i] = vtol.state[3:6]
        t = i * dt
        T = m * g * 1.02
        if t < 5:
            tilt = 0.0
            tx, ty, tz = 0.0, 0.1, 0.0
        elif t < 10:
            blend = (t - 5) / 5.0
            tilt = blend * np.pi / 3
            T = m * g / max(np.cos(tilt), 0.3)
            tx, ty, tz = 0.0, 0.0, 0.0
        elif t < 15:
            tilt = np.pi / 3
            T = m * g / max(np.cos(tilt), 0.3) * 0.95
            tx, ty, tz = 0.0, -0.05, 0.0
        else:
            blend = (t - 15) / 5.0
            tilt = np.pi / 3 * (1 - blend)
            T = m * g * 1.02
            tx, ty, tz = 0.0, 0.0, 0.0

        tilt_angles[i] = tilt
        vtol.step(np.array([T, tx, ty, tz, tilt]), dt)

        if np.any(np.isnan(vtol.state)):
            positions = positions[:i]
            eulers = eulers[:i]
            tilt_angles = tilt_angles[:i]
            break

    n_steps = len(positions)
    if n_steps < 2:
        print("VTOL simulation too short — skipping")
        return

    viz = ThreePanelViz(title="VTOL Hover → Cruise Transition", world_size=WORLD_SIZE)

    # Inset: tilt angle over time
    times = np.arange(n_steps) * dt
    ax_tilt = viz.fig.add_axes([0.60, 0.03, 0.36, 0.18])
    ax_tilt.set_xlim(0, times[-1])
    ax_tilt.set_ylim(-5, 65)
    ax_tilt.set_xlabel("Time [s]", fontsize=7)
    ax_tilt.set_ylabel("Tilt [deg]", fontsize=7)
    ax_tilt.tick_params(labelsize=6)
    ax_tilt.grid(True, alpha=0.2)
    (tilt_line,) = ax_tilt.plot([], [], "darkorange", lw=0.8)

    anim = SimAnimator("vtol_transition", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="dodgerblue")

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    vehicle_arts_3d: list = []

    def update(f: int) -> None:
        k = idx[min(f, len(idx) - 1)]
        viz.update_trail(trail_arts, positions, k)

        clear_vehicle_artists(vehicle_arts_3d)
        R = Quadrotor.rotation_matrix(*eulers[k])
        vehicle_arts_3d.extend(draw_quadrotor_3d(viz.ax3d, positions[k], R, size=1.5))

        clear_vehicle_artists(viz._vehicle_arts_top)
        (dt_t,) = viz.ax_top.plot(positions[k, 0], positions[k, 1], "ko", ms=5, zorder=5)
        viz._vehicle_arts_top.append(dt_t)
        clear_vehicle_artists(viz._vehicle_arts_side)
        (dt_s,) = viz.ax_side.plot(positions[k, 0], positions[k, 2], "ko", ms=5, zorder=5)
        viz._vehicle_arts_side.append(dt_s)

        tilt_line.set_data(times[:k], np.degrees(tilt_angles[:k]))

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
