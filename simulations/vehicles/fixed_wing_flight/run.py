# Erwin Lejeune - 2026-02-17
"""Fixed-wing level flight and gentle banked turn demonstration.

Reference: R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and
Practice," Princeton University Press, 2012.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.vehicles.fixed_wing import FixedWing
from uav_sim.visualization import SimAnimator


def main() -> None:
    fw = FixedWing()
    state = np.zeros(12)
    state[2] = 100.0  # altitude
    state[6] = 15.0  # forward speed u
    fw.reset(state=state)

    dt, duration = 0.002, 20.0
    steps = int(duration / dt)
    positions = np.zeros((steps, 3))

    for i in range(steps):
        positions[i] = fw.state[:3]
        t = i * dt
        throttle = 0.5
        if t < 5:
            elevator, aileron, rudder = -0.01, 0.0, 0.0
        elif t < 12:
            elevator, aileron, rudder = -0.01, 0.05, 0.01
        else:
            elevator, aileron, rudder = -0.01, -0.03, -0.005
        fw.step(np.array([elevator, aileron, rudder, throttle]), dt)
        if np.any(np.isnan(fw.state)):
            positions = positions[:i]
            break

    anim = SimAnimator("fixed_wing_flight", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Fixed-Wing Flight")
    (trail,) = ax.plot([], [], [], "b-", lw=1.5)
    (dot,) = ax.plot([], [], [], "ro", ms=6)
    ax.scatter(*positions[0], c="g", s=80, marker="^", label="Start")
    anim.set_equal_3d(ax, positions, pad=10.0)
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
