# Erwin Lejeune - 2026-02-15
"""Complementary filter attitude estimation from noisy sensors.

Reference: R. Mahony et al., "Nonlinear Complementary Filters on the Special
Orthogonal Group," IEEE TAC, 2008. DOI: 10.1109/TAC.2008.923738
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.estimation.complementary_filter import ComplementaryFilter
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    dt, N = 0.01, 500
    rng = np.random.default_rng(42)
    cf = ComplementaryFilter(alpha=0.98)
    tr, tp, er, ep = [np.zeros(N) for _ in range(4)]
    for i in range(N):
        t = i * dt
        roll = 0.3 * np.sin(0.5 * t)
        pitch = 0.2 * np.cos(0.3 * t)
        tr[i] = roll
        tp[i] = pitch
        gyro = np.array(
            [
                0.15 * np.cos(0.5 * t) + rng.normal(0, 0.05),
                -0.06 * np.sin(0.3 * t) + rng.normal(0, 0.05),
                rng.normal(0, 0.02),
            ]
        )
        g = 9.81
        accel = np.array(
            [
                -g * np.sin(pitch) + rng.normal(0, 0.3),
                g * np.sin(roll) * np.cos(pitch) + rng.normal(0, 0.3),
                g * np.cos(roll) * np.cos(pitch) + rng.normal(0, 0.3),
            ]
        )
        est = cf.update(gyro, accel, dt)
        er[i] = est[0]
        ep[i] = est[1]
    times = np.arange(N) * dt
    skip = max(1, N // 200)
    idx = list(range(0, N, skip))
    anim = SimAnimator("complementary_filter", out_dir=Path(__file__).parent)
    _, axes = anim.figure_2d("Complementary Filter", nrows=2, sharex=True)
    (lt_r,) = axes[0].plot([], [], "k-", lw=1.5, label="True")
    (le_r,) = axes[0].plot([], [], "b-", lw=1.2, label="Est")
    axes[0].set_xlim(0, N * dt)
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_ylabel("Roll [rad]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    (lt_p,) = axes[1].plot([], [], "k-", lw=1.5, label="True")
    (le_p,) = axes[1].plot([], [], "b-", lw=1.2, label="Est")
    axes[1].set_xlim(0, N * dt)
    axes[1].set_ylim(-0.4, 0.4)
    axes[1].set_ylabel("Pitch [rad]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    def update(f):
        k = idx[f]
        lt_r.set_data(times[:k], tr[:k])
        le_r.set_data(times[:k], er[:k])
        lt_p.set_data(times[:k], tp[:k])
        le_p.set_data(times[:k], ep[:k])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
