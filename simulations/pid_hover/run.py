# Erwin Lejeune - 2026-02-15
"""PID-controlled hover: take-off to 1 m and hold.

Reference: L. R. G. Carrillo et al., "Quad Rotorcraft Control," Springer, 2013.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.control.pid_controller import CascadedPIDController
from quadrotor_sim.models.quadrotor import Quadrotor
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    ctrl = CascadedPIDController()
    target = np.array([0.0, 0.0, 1.0])
    dt, duration = 0.002, 5.0
    steps = int(duration / dt)
    states = np.zeros((steps, 12))
    for i in range(steps):
        states[i] = quad.state
        quad.step(ctrl.compute(quad.state, target, dt=dt), dt)
    pos = states[:, :3]
    skip = max(1, len(pos) // 200)
    idx = list(range(0, len(pos), skip))
    anim = SimAnimator("pid_hover", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("PID Hover")
    (trail,) = ax.plot([], [], [], "b-", lw=1.2, alpha=0.6)
    (dot,) = ax.plot([], [], [], "ro", ms=6)
    ax.scatter(*target, c="g", s=80, marker="*", label="Target", zorder=5)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.1, 1.3)
    ax.legend(loc="upper left")

    def update(f):
        k = idx[f]
        trail.set_data(pos[:k, 0], pos[:k, 1])
        trail.set_3d_properties(pos[:k, 2])
        dot.set_data([pos[k, 0]], [pos[k, 1]])
        dot.set_3d_properties([pos[k, 2]])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
