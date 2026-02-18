# Erwin Lejeune - 2026-02-17
"""VTOL tilt-rotor transition: hover → cruise → hover.

Reference: R. Bapst et al., "Design and Implementation of an Unmanned
Tail-Sitter," IROS, 2015.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.vehicles.vtol import Tiltrotor
from uav_sim.visualization import SimAnimator


def main() -> None:
    vtol = Tiltrotor()
    state = np.zeros(12)
    state[2] = 20.0  # altitude
    vtol.reset(state=state)

    dt, duration = 0.005, 20.0
    steps = int(duration / dt)
    positions = np.zeros((steps, 3))
    tilt_angles = np.zeros(steps)

    g = vtol.vtol_params.gravity
    m = vtol.vtol_params.mass

    for i in range(steps):
        positions[i] = vtol.state[:3]
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

    anim = SimAnimator("vtol_transition", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("VTOL Hover → Cruise Transition")
    (trail,) = ax.plot([], [], [], "b-", lw=1.5)
    (dot,) = ax.plot([], [], [], "ro", ms=6)
    ax.scatter(*positions[0], c="g", s=80, marker="^", label="Start")
    anim.set_equal_3d(ax, positions, pad=5.0)
    ax.legend()

    skip = max(1, len(positions) // 200)
    idx = list(range(0, len(positions), skip))

    def update(f):
        k = idx[min(f, len(idx) - 1)]
        trail.set_data(positions[:k, 0], positions[:k, 1])
        trail.set_3d_properties(positions[:k, 2])
        dot.set_data([positions[k, 0]], [positions[k, 1]])
        dot.set_3d_properties([positions[k, 2]])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
