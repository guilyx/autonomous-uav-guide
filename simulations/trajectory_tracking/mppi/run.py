# Erwin Lejeune - 2026-02-17
"""MPPI trajectory tracking: sampled rollouts + quadrotor execution.

Phase 1 — Algorithm: shows sampled trajectory rollouts and the weighted mean
           trajectory converging on the target at each MPPI step.
Phase 2 — Platform: quadrotor executing the MPPI-computed commands with full
           dynamics, alongside live state history plots.

Reference: G. Williams et al., "Information Theoretic MPC for Model-Based
Reinforcement Learning," ICRA, 2017. DOI: 10.1109/ICRA.2017.7989202
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.trajectory_tracking.mppi import MPPITracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")


def _dyn(x, u, dt):
    pos, vel = x[:3], x[3:6]
    nv = vel + u * dt
    np_ = pos + nv * dt
    return np.concatenate([np_, nv])


def _cost(x, _u, ref):
    if ref is None:
        return 0.0
    return float(
        np.sum((x[:3] - ref[:3]) ** 2) + 0.1 * np.sum((x[3:6] - ref[3:6]) ** 2)
    )


def main() -> None:
    tgt = np.array([2.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    n_samples = 64
    horizon = 15
    tracker = MPPITracker(
        state_dim=6,
        control_dim=3,
        horizon=horizon,
        num_samples=n_samples,
        lambda_=1.0,
        control_std=np.array([2.0, 2.0, 2.0]),
        dynamics=_dyn,
        cost_fn=_cost,
        dt=0.05,
    )

    # ── Phase 1: run MPPI steps (point-mass) recording rollouts ───────────
    state = np.zeros(6)
    dt_mppi, n_steps = 0.05, 150
    mppi_hist = np.zeros((n_steps, 6))
    mppi_controls = np.zeros((n_steps, 3))
    for i in range(n_steps):
        u = tracker.compute(state, reference=tgt, seed=i)
        mppi_controls[i] = u
        state = _dyn(state, u, dt_mppi)
        mppi_hist[i] = state

    # ── Phase 2: quadrotor executing the commands ─────────────────────────
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    ctrl = CascadedPIDController()
    dt_fly = 0.005
    flight_pos: list[np.ndarray] = []
    flight_euler: list[np.ndarray] = []
    for i in range(n_steps):
        wp = mppi_hist[i, :3]
        for _ in range(int(dt_mppi / dt_fly)):
            s = quad.state
            p = s[:3].copy()
            if np.any(np.isnan(p)) or np.any(np.abs(p) > 50):
                break
            flight_pos.append(p)
            flight_euler.append(s[3:6].copy())
            quad.step(ctrl.compute(s, wp, dt=dt_fly), dt_fly)
    flight_pos_arr = np.array(flight_pos) if flight_pos else mppi_hist[:1, :3]
    flight_euler_arr = np.array(flight_euler) if flight_euler else np.zeros((1, 3))

    # ── Animation ─────────────────────────────────────────────────────────
    t_mppi = np.arange(n_steps) * dt_mppi
    mppi_skip = max(1, n_steps // 120)
    mppi_idx = list(range(0, n_steps, mppi_skip))
    fly_skip = max(1, len(flight_pos_arr) // 120)
    fly_idx = list(range(0, len(flight_pos_arr), fly_skip))
    n_mf = len(mppi_idx)
    n_ff = len(fly_idx)
    total = n_mf + n_ff

    anim = SimAnimator("mppi", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 7))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1])
    fig.suptitle("MPPI Trajectory Tracking", fontsize=13)

    # 3D view
    ax3d.scatter(*tgt[:3], c="g", s=120, marker="*", label="Target", zorder=5)
    ax3d.scatter(0, 0, 0, c="blue", s=60, marker="o", label="Start", zorder=5)
    ax3d.set_xlim(-0.5, 3)
    ax3d.set_ylim(-0.5, 3)
    ax3d.set_zlim(-0.5, 2)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (mppi_trail,) = ax3d.plot([], [], [], "c-", lw=1.5, alpha=0.7, label="MPPI")
    (mppi_dot,) = ax3d.plot([], [], [], "co", ms=5)
    (fly_trail,) = ax3d.plot([], [], [], "orange", lw=1.8, label="Quad")
    (fly_dot,) = ax3d.plot([], [], [], "ko", ms=7)

    # Position subplot
    ax_pos.set_xlim(0, n_steps * dt_mppi)
    ax_pos.set_ylim(-0.5, 3)
    ax_pos.set_ylabel("Pos [m]", fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    for j, v in enumerate(tgt[:3]):
        ax_pos.axhline(v, color=f"C{j}", ls="--", alpha=0.3)
    lbl = ["x", "y", "z"]
    lp = [ax_pos.plot([], [], f"C{j}-", lw=1, label=lb)[0] for j, lb in enumerate(lbl)]
    ax_pos.legend(fontsize=6, ncol=3, loc="upper right")
    ax_pos.tick_params(labelsize=7)

    # Velocity subplot
    ax_vel.set_xlim(0, n_steps * dt_mppi)
    ax_vel.set_ylim(-2, 3)
    ax_vel.set_xlabel("Time [s]", fontsize=8)
    ax_vel.set_ylabel("Vel [m/s]", fontsize=8)
    ax_vel.grid(True, alpha=0.3)
    lv = [
        ax_vel.plot([], [], f"C{j}-", lw=1, label=f"v{lb}")[0]
        for j, lb in enumerate(lbl)
    ]
    ax_vel.legend(fontsize=6, ncol=3, loc="upper right")
    ax_vel.tick_params(labelsize=7)

    title = ax3d.set_title("Phase 1: MPPI Planning")
    vehicle_arts: list = []

    def update(f):
        if f < n_mf:
            k = mppi_idx[f]
            mppi_trail.set_data(mppi_hist[: k + 1, 0], mppi_hist[: k + 1, 1])
            mppi_trail.set_3d_properties(mppi_hist[: k + 1, 2])
            mppi_dot.set_data([mppi_hist[k, 0]], [mppi_hist[k, 1]])
            mppi_dot.set_3d_properties([mppi_hist[k, 2]])
            for j, line in enumerate(lp):
                line.set_data(t_mppi[: k + 1], mppi_hist[: k + 1, j])
            for j, line in enumerate(lv):
                line.set_data(t_mppi[: k + 1], mppi_hist[: k + 1, 3 + j])
            title.set_text(f"Phase 1: MPPI Planning — step {k + 1}/{n_steps}")
        else:
            fi = f - n_mf
            k = fly_idx[min(fi, len(fly_idx) - 1)]
            mppi_dot.set_data([], [])
            mppi_dot.set_3d_properties([])
            fly_trail.set_data(flight_pos_arr[:k, 0], flight_pos_arr[:k, 1])
            fly_trail.set_3d_properties(flight_pos_arr[:k, 2])
            fly_dot.set_data([flight_pos_arr[k, 0]], [flight_pos_arr[k, 1]])
            fly_dot.set_3d_properties([flight_pos_arr[k, 2]])
            clear_vehicle_artists(vehicle_arts)
            R = Quadrotor.rotation_matrix(*flight_euler_arr[k])
            vehicle_arts.extend(
                draw_quadrotor_3d(ax3d, flight_pos_arr[k], R, size=0.15)
            )
            title.set_text("Phase 2: Quadrotor Executing MPPI Trajectory")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
