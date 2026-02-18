# Erwin Lejeune - 2026-02-17
"""GPS + IMU sensor fusion via EKF for quadrotor localisation.

Demonstrates the benefit of fusing high-rate IMU predictions with low-rate
GPS corrections.  The quad flies a figure-8 via pure pursuit.  Three
estimates are compared: IMU-only (dead-reckoning), GPS-only, and EKF fusion.

Left panel: 2D trajectory comparison with quadrotor model.
Right panels: position error and covariance trace over time.

Reference: S. Thrun et al., "Probabilistic Robotics," MIT Press, 2005, Ch. 3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from uav_sim.estimation.ekf import ExtendedKalmanFilter
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
)

matplotlib.use("Agg")

# Sensor noise parameters
GPS_STD = 0.6
IMU_ACCEL_STD = 0.15
GPS_RATE_HZ = 5  # GPS updates at 5 Hz


def main() -> None:
    rng = np.random.default_rng(42)

    # ── Figure-8 flight path ──────────────────────────────────────────────
    n_wp = 60
    t_wp = np.linspace(0, 2 * np.pi, n_wp, endpoint=False)
    radius = 2.0
    cruise_alt = 1.0
    path_3d = np.column_stack(
        [
            radius * np.sin(t_wp),
            radius * np.sin(2 * t_wp) / 2,
            np.full(n_wp, cruise_alt),
        ]
    )

    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, cruise_alt]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=0.5, waypoint_threshold=0.2, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=40.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    n_steps = len(flight_states)
    dt = 0.005

    # ── EKF setup: state = [x, y, vx, vy] ────────────────────────────────
    def _f(x, u, dt_):
        return np.array(
            [x[0] + x[2] * dt_, x[1] + x[3] * dt_, x[2] + u[0] * dt_, x[3] + u[1] * dt_]
        )

    def _F(_x, _u, dt_):
        return np.array([[1, 0, dt_, 0], [0, 1, 0, dt_], [0, 0, 1, 0], [0, 0, 0, 1]])

    def _h(x):
        return x[:2]

    def _H(_x):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    ekf = ExtendedKalmanFilter(state_dim=4, meas_dim=2, f=_f, h=_h, F_jac=_F, H_jac=_H)
    ekf.Q = np.diag([0.001, 0.001, IMU_ACCEL_STD**2, IMU_ACCEL_STD**2])
    ekf.R = np.diag([GPS_STD**2, GPS_STD**2])
    ekf.x = np.array([0.0, 0.0, 0.0, 0.0])
    ekf.P = np.eye(4) * 0.5

    gps_period = int(1.0 / (GPS_RATE_HZ * dt))

    true_xy = np.zeros((n_steps, 2))
    ekf_xy = np.zeros((n_steps, 2))
    gps_xy_list: list[np.ndarray] = []
    imu_xy = np.zeros((n_steps, 2))
    cov_trace = np.zeros(n_steps)
    imu_pos = np.array([0.0, 0.0])

    for i in range(n_steps):
        s = flight_states[i]
        true_xy[i] = s[:2]

        # IMU dead-reckoning
        accel_noisy = s[6:8] + rng.normal(0, IMU_ACCEL_STD, 2)
        if i > 0:
            imu_pos = imu_pos + accel_noisy * dt
        imu_xy[i] = imu_pos

        # EKF predict with IMU
        u_imu = accel_noisy
        ekf.predict(u_imu, dt)

        # GPS update at low rate
        if i % gps_period == 0:
            gps_meas = s[:2] + rng.normal(0, GPS_STD, 2)
            ekf.update(gps_meas)
            gps_xy_list.append(gps_meas.copy())

        ekf_xy[i] = ekf.x[:2]
        cov_trace[i] = np.trace(ekf.P[:2, :2])

    gps_xy = np.array(gps_xy_list) if gps_xy_list else np.zeros((1, 2))
    times = np.arange(n_steps) * dt
    err_ekf = np.sqrt(np.sum((true_xy - ekf_xy) ** 2, axis=1))
    err_imu = np.sqrt(np.sum((true_xy - imu_xy) ** 2, axis=1))

    # ── Animation ─────────────────────────────────────────────────────────
    skip = max(1, n_steps // 100)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    anim = SimAnimator("gps_imu_fusion", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 6.5))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_cov = fig.add_subplot(gs[1, 1])
    fig.suptitle("GPS + IMU Fusion (EKF)", fontsize=13)

    ax_map.set_aspect("equal")
    pad = 1.5
    ax_map.set_xlim(true_xy[:, 0].min() - pad, true_xy[:, 0].max() + pad)
    ax_map.set_ylim(true_xy[:, 1].min() - pad, true_xy[:, 1].max() + pad)
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.grid(True, alpha=0.2)
    (true_line,) = ax_map.plot([], [], "k-", lw=1.5, label="True")
    (ekf_line,) = ax_map.plot([], [], "b-", lw=1.0, label="EKF")
    (imu_line,) = ax_map.plot([], [], "r--", lw=0.8, alpha=0.5, label="IMU only")
    ax_map.plot(gps_xy[:, 0], gps_xy[:, 1], "g.", ms=3, alpha=0.3, label="GPS")
    ell = Ellipse((0, 0), 0, 0, fill=False, color="blue", lw=1, ls="--")
    ax_map.add_patch(ell)
    ax_map.legend(fontsize=7, loc="upper left")

    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(0, max(1.0, max(err_ekf.max(), err_imu.max()) * 1.1))
    ax_err.set_ylabel("Position Error [m]", fontsize=8)
    ax_err.grid(True, alpha=0.3)
    ax_err.tick_params(labelsize=7)
    (err_ekf_line,) = ax_err.plot([], [], "b-", lw=1, label="EKF")
    (err_imu_line,) = ax_err.plot([], [], "r--", lw=0.8, alpha=0.5, label="IMU")
    ax_err.legend(fontsize=6)

    ax_cov.set_xlim(0, times[-1])
    ax_cov.set_ylim(0, max(0.5, cov_trace.max() * 1.1))
    ax_cov.set_xlabel("Time [s]", fontsize=8)
    ax_cov.set_ylabel("tr(P_pos) [m²]", fontsize=8)
    ax_cov.grid(True, alpha=0.3)
    ax_cov.tick_params(labelsize=7)
    (cov_line,) = ax_cov.plot([], [], "b-", lw=1)

    vehicle_arts: list = []

    def update(f):
        k = idx[f]
        true_line.set_data(true_xy[:k, 0], true_xy[:k, 1])
        ekf_line.set_data(ekf_xy[:k, 0], ekf_xy[:k, 1])
        imu_line.set_data(imu_xy[:k, 0], imu_xy[:k, 1])
        err_ekf_line.set_data(times[:k], err_ekf[:k])
        err_imu_line.set_data(times[:k], err_imu[:k])
        cov_line.set_data(times[:k], cov_trace[:k])

        # Covariance ellipse
        P = ekf.P[:2, :2]
        vals, vecs = np.linalg.eigh(P)
        vals = np.maximum(vals, 1e-8)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ell.set_center(ekf_xy[k])
        ell.width = 4 * np.sqrt(vals[0])
        ell.height = 4 * np.sqrt(vals[1])
        ell.angle = angle

        clear_vehicle_artists(vehicle_arts)
        vehicle_arts.extend(draw_quadrotor_2d(ax_map, true_xy[k], flight_states[k, 5], size=0.25))

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
