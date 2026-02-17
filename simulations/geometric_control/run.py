# Erwin Lejeune - 2026-02-15
"""Geometric SO(3) controller: attitude recovery to hover.

Reference: T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of
a Quadrotor UAV on SE(3)," CDC, 2010. DOI: 10.1109/CDC.2010.5717652
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.control.geometric_controller import GeometricController
from quadrotor_sim.models.quadrotor import Quadrotor
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 1.0]), euler=np.array([0.3, -0.2, 0.0]))
    hover_f = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hover_f))
    ctrl = GeometricController()
    target = np.array([0.0, 0.0, 1.0])
    dt, dur = 0.002, 4.0
    steps = int(dur / dt)
    times = np.zeros(steps)
    states = np.zeros((steps, 12))
    for i in range(steps):
        times[i] = quad.time
        states[i] = quad.state
        quad.step(ctrl.compute(quad.state, target), dt)
    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
    anim = SimAnimator("geometric_control", out_dir=Path(__file__).parent)
    _, axes = anim.figure_2d("Geometric SO(3) — Attitude Recovery", nrows=2, sharex=True)
    labels_e = ["roll", "pitch", "yaw"]
    le = [axes[0].plot([], [], label=lb)[0] for lb in labels_e]
    axes[0].set_xlim(0, dur)
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_ylabel("Euler [rad]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    labels_p = ["x", "y", "z"]
    lp = [axes[1].plot([], [], label=lb)[0] for lb in labels_p]
    axes[1].set_xlim(0, dur)
    axes[1].set_ylim(-0.2, 1.3)
    axes[1].set_ylabel("Pos [m]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    def update(f):
        k = idx[f]
        for j, line in enumerate(le):
            line.set_data(times[:k], states[:k, 3 + j])
        for j, line in enumerate(lp):
            line.set_data(times[:k], states[:k, j])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
