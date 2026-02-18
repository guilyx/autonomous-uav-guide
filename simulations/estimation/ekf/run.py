# Erwin Lejeune - 2026-02-17
"""EKF localisation on a quadrotor: live covariance ellipses + state history.

Phase 1 — Algorithm: step-by-step predict/update showing growing covariance
           ellipse, measurement scatter, and true vs estimated state.
Phase 2 — Platform: quadrotor flying a circle with noisy GPS, EKF fusing.

Reference: S. Thrun et al., "Probabilistic Robotics," MIT Press, 2005, Ch. 3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from uav_sim.estimation.ekf import ExtendedKalmanFilter
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.sensors.gps import GPS
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
)

matplotlib.use("Agg")


def main() -> None:
    # ── quadrotor flying a circle with GPS + EKF ──────────────────────────
    quad = Quadrotor()
    quad.reset(position=np.array([3.0, 0.0, 2.0]))
    ctrl = CascadedPIDController()
    gps = GPS(noise_std=0.4, seed=42)

    def _f(x, _u, dt):
        return np.array([x[0] + x[2] * dt, x[1] + x[3] * dt, x[2], x[3]])

    def _F(_x, _u, dt):
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    def _h(x):
        return x[:2]

    def _H(_x):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    ekf = ExtendedKalmanFilter(state_dim=4, meas_dim=2, f=_f, h=_h, F_jac=_F, H_jac=_H)
    ekf.Q = np.diag([0.02, 0.02, 0.1, 0.1])
    ekf.R = np.diag([0.16, 0.16])
    ekf.x = np.array([3.0, 0.0, 0.0, 0.0])
    ekf.P = np.eye(4) * 0.5

    dt, duration = 0.02, 10.0
    steps = int(duration / dt)
    true_xy = np.zeros((steps, 2))
    est_xy = np.zeros((steps, 2))
    meas_xy = np.zeros((steps, 2))
    cov_history = np.zeros((steps, 2, 2))
    times = np.arange(steps) * dt
    radius = 3.0

    yaw_hist = np.zeros(steps)
    for i in range(steps):
        t = i * dt
        target = np.array([radius * np.cos(0.5 * t), radius * np.sin(0.5 * t), 2.0])
        quad.step(ctrl.compute(quad.state, target, dt=dt), dt)
        pos = quad.state[:3]
        true_xy[i] = pos[:2]
        yaw_hist[i] = quad.state[5]

        gps_meas = gps.sense(quad.state)
        meas_xy[i] = gps_meas[:2]

        ekf.predict(np.zeros(2), dt)
        ekf.update(gps_meas[:2])
        est_xy[i] = ekf.x[:2]
        cov_history[i] = ekf.P[:2, :2]

    # ── animation ─────────────────────────────────────────────────────────
    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    anim = SimAnimator("ekf", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(13, 6))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_cov = fig.add_subplot(gs[1, 1])
    fig.suptitle("EKF Localisation — GPS Fusion", fontsize=13)

    # Map view
    ax_map.set_aspect("equal")
    ax_map.set_xlim(-5, 5)
    ax_map.set_ylim(-5, 5)
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.grid(True, alpha=0.2)
    (true_line,) = ax_map.plot([], [], "k-", lw=1.5, label="True")
    (est_line,) = ax_map.plot([], [], "b-", lw=1.2, label="EKF")
    (meas_scat,) = ax_map.plot([], [], "r.", ms=3, alpha=0.3, label="GPS")
    (true_dot,) = ax_map.plot([], [], "ko", ms=6)
    (est_dot,) = ax_map.plot([], [], "bs", ms=5)
    ellipse_patch = Ellipse((0, 0), 0, 0, angle=0, fill=False, color="blue", lw=1.5, ls="--")
    ax_map.add_patch(ellipse_patch)
    ax_map.legend(fontsize=7, loc="upper left")

    # Error subplot
    err = np.sqrt((true_xy[:, 0] - est_xy[:, 0]) ** 2 + (true_xy[:, 1] - est_xy[:, 1]) ** 2)
    ax_err.set_xlim(0, duration)
    ax_err.set_ylim(0, max(1.0, err.max() * 1.1))
    ax_err.set_ylabel("Position Error [m]", fontsize=8)
    ax_err.grid(True, alpha=0.3)
    (err_line,) = ax_err.plot([], [], "r-", lw=1)
    ax_err.tick_params(labelsize=7)

    # Covariance trace subplot
    cov_trace = np.array([np.trace(cov_history[i]) for i in range(steps)])
    ax_cov.set_xlim(0, duration)
    ax_cov.set_ylim(0, max(0.5, cov_trace.max() * 1.1))
    ax_cov.set_xlabel("Time [s]", fontsize=8)
    ax_cov.set_ylabel("tr(P) [m²]", fontsize=8)
    ax_cov.grid(True, alpha=0.3)
    (cov_line,) = ax_cov.plot([], [], "b-", lw=1)
    ax_cov.tick_params(labelsize=7)

    def _draw_ellipse(mean, cov, patch, n_std=2.0):
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * n_std * np.sqrt(vals)
        patch.set_center(mean)
        patch.width = w
        patch.height = h
        patch.angle = angle

    vehicle_arts: list = []

    def update(f):
        k = idx[f]
        true_line.set_data(true_xy[:k, 0], true_xy[:k, 1])
        est_line.set_data(est_xy[:k, 0], est_xy[:k, 1])
        meas_scat.set_data(meas_xy[:k, 0], meas_xy[:k, 1])
        true_dot.set_data([true_xy[k, 0]], [true_xy[k, 1]])
        est_dot.set_data([est_xy[k, 0]], [est_xy[k, 1]])
        _draw_ellipse(est_xy[k], cov_history[k], ellipse_patch)
        err_line.set_data(times[:k], err[:k])
        cov_line.set_data(times[:k], cov_trace[:k])
        clear_vehicle_artists(vehicle_arts)
        vehicle_arts.extend(draw_quadrotor_2d(ax_map, true_xy[k], yaw_hist[k], size=0.3))

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
