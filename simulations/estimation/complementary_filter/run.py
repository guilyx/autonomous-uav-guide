# Erwin Lejeune - 2026-02-17
"""Complementary filter: multi-panel showing gyro, accel, and fused estimates.

Shows the raw sensor contributions (gyro integration vs accelerometer)
alongside the fused estimate, demonstrating the complementary weighting.

Reference: R. Mahony et al., "Nonlinear Complementary Filters on the Special
Orthogonal Group," IEEE TAC, 2008. DOI: 10.1109/TAC.2008.923738
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.estimation.complementary_filter import ComplementaryFilter
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def main() -> None:
    dt, N = 0.01, 500
    rng = np.random.default_rng(42)
    cf = ComplementaryFilter(alpha=0.98)

    tr, tp = np.zeros(N), np.zeros(N)
    er, ep = np.zeros(N), np.zeros(N)
    gyro_r, gyro_p = np.zeros(N), np.zeros(N)
    accel_r, accel_p = np.zeros(N), np.zeros(N)
    err_roll, err_pitch = np.zeros(N), np.zeros(N)

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

        accel_roll_meas = np.arctan2(accel[1], accel[2])
        accel_pitch_meas = np.arctan2(-accel[0], np.sqrt(accel[1] ** 2 + accel[2] ** 2))
        accel_r[i] = accel_roll_meas
        accel_p[i] = accel_pitch_meas

        if i > 0:
            gyro_r[i] = er[i - 1] + gyro[0] * dt
            gyro_p[i] = ep[i - 1] + gyro[1] * dt

        est = cf.update(gyro, accel, dt)
        er[i] = est[0]
        ep[i] = est[1]
        err_roll[i] = abs(roll - est[0])
        err_pitch[i] = abs(pitch - est[1])

    times = np.arange(N) * dt
    skip = max(1, N // 200)
    idx = list(range(0, N, skip))
    n_frames = len(idx)

    anim = SimAnimator("complementary_filter", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(13, 8))
    anim._fig = fig
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)
    ax_roll = fig.add_subplot(gs[0, 0])
    ax_pitch = fig.add_subplot(gs[0, 1])
    ax_src_r = fig.add_subplot(gs[1, 0])
    ax_src_p = fig.add_subplot(gs[1, 1])
    ax_err = fig.add_subplot(gs[2, :])
    fig.suptitle("Complementary Filter — Sensor Fusion", fontsize=13)

    for ax in [ax_roll, ax_pitch, ax_src_r, ax_src_p, ax_err]:
        ax.set_xlim(0, N * dt)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    ax_roll.set_ylim(-0.5, 0.5)
    ax_roll.set_ylabel("Roll [rad]", fontsize=8)
    (ltr,) = ax_roll.plot([], [], "k-", lw=1.5, label="True")
    (ler,) = ax_roll.plot([], [], "b-", lw=1.2, label="CF")
    ax_roll.legend(fontsize=6, loc="upper right")

    ax_pitch.set_ylim(-0.4, 0.4)
    ax_pitch.set_ylabel("Pitch [rad]", fontsize=8)
    (ltp,) = ax_pitch.plot([], [], "k-", lw=1.5, label="True")
    (lep,) = ax_pitch.plot([], [], "b-", lw=1.2, label="CF")
    ax_pitch.legend(fontsize=6, loc="upper right")

    ax_src_r.set_ylim(-0.8, 0.8)
    ax_src_r.set_ylabel("Roll Sources [rad]", fontsize=8)
    (lgr,) = ax_src_r.plot([], [], "r-", lw=0.8, alpha=0.5, label="Gyro int.")
    (lar,) = ax_src_r.plot([], [], "g-", lw=0.8, alpha=0.5, label="Accel")
    ax_src_r.legend(fontsize=6, loc="upper right")

    ax_src_p.set_ylim(-0.6, 0.6)
    ax_src_p.set_ylabel("Pitch Sources [rad]", fontsize=8)
    (lgp,) = ax_src_p.plot([], [], "r-", lw=0.8, alpha=0.5, label="Gyro int.")
    (lap,) = ax_src_p.plot([], [], "g-", lw=0.8, alpha=0.5, label="Accel")
    ax_src_p.legend(fontsize=6, loc="upper right")

    ax_err.set_ylim(0, 0.15)
    ax_err.set_xlabel("Time [s]", fontsize=8)
    ax_err.set_ylabel("|Error| [rad]", fontsize=8)
    (le_r_err,) = ax_err.plot([], [], "r-", lw=1, label="|Δroll|")
    (le_p_err,) = ax_err.plot([], [], "b-", lw=1, label="|Δpitch|")
    ax_err.legend(fontsize=6, ncol=2, loc="upper right")

    def update(f):
        k = idx[f]
        ltr.set_data(times[:k], tr[:k])
        ler.set_data(times[:k], er[:k])
        ltp.set_data(times[:k], tp[:k])
        lep.set_data(times[:k], ep[:k])
        lgr.set_data(times[:k], gyro_r[:k])
        lar.set_data(times[:k], accel_r[:k])
        lgp.set_data(times[:k], gyro_p[:k])
        lap.set_data(times[:k], accel_p[:k])
        le_r_err.set_data(times[:k], err_roll[:k])
        le_p_err.set_data(times[:k], err_pitch[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
