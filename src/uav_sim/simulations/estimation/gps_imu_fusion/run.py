# Erwin Lejeune - 2026-02-18
"""GPS + IMU sensor fusion via EKF for quadrotor localisation.

Three panels (3D, top-down, side) showing the quadrotor flying a figure-8
through a 30 m urban environment.  Three trajectory estimates are compared:
IMU-only dead-reckoning, raw GPS fixes, and the fused EKF output.

An inset panel shows position error and covariance trace over time.

Reference: S. Thrun et al., "Probabilistic Robotics," MIT Press, 2005, Ch. 3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.estimation.ekf import ExtendedKalmanFilter
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0
GPS_STD = 0.5
IMU_ACCEL_STD = 0.10
GPS_RATE_HZ = 5


def main() -> None:
    rng = np.random.default_rng(42)

    world, buildings = default_world()

    cx, cy = 15.0, 15.0
    n_wp = 80
    t_wp = np.linspace(0, 2 * np.pi, n_wp, endpoint=False)
    radius = 8.0
    path_3d = np.column_stack(
        [
            cx + radius * np.sin(t_wp),
            cy + radius * np.sin(2 * t_wp) / 2,
            np.full(n_wp, CRUISE_ALT),
        ]
    )

    quad = Quadrotor()
    quad.reset(position=np.array([path_3d[0, 0], path_3d[0, 1], CRUISE_ALT]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=60.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    n_steps = len(flight_states)
    dt = 0.005

    # ── EKF setup: state = [x, y, z, vx, vy, vz] ──────────────────────
    def _f(x, u, dt_):
        return np.array(
            [
                x[0] + x[3] * dt_,
                x[1] + x[4] * dt_,
                x[2] + x[5] * dt_,
                x[3] + u[0] * dt_,
                x[4] + u[1] * dt_,
                x[5] + u[2] * dt_,
            ]
        )

    def _F(_x, _u, dt_):
        F = np.eye(6)
        F[0, 3] = dt_
        F[1, 4] = dt_
        F[2, 5] = dt_
        return F

    def _h(x):
        return x[:3]

    def _H(_x):
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        return H

    ekf = ExtendedKalmanFilter(state_dim=6, meas_dim=3, f=_f, h=_h, F_jac=_F, H_jac=_H)
    ekf.Q = np.diag([0.001, 0.001, 0.001, IMU_ACCEL_STD**2, IMU_ACCEL_STD**2, IMU_ACCEL_STD**2])
    ekf.R = np.diag([GPS_STD**2, GPS_STD**2, GPS_STD**2])
    ekf.x = np.zeros(6)
    ekf.x[:3] = flight_states[0, :3]
    ekf.P = np.eye(6) * 0.5

    gps_period = int(1.0 / (GPS_RATE_HZ * dt))

    true_xyz = np.zeros((n_steps, 3))
    ekf_xyz = np.zeros((n_steps, 3))
    imu_xyz = np.zeros((n_steps, 3))
    gps_xyz_list: list[np.ndarray] = []
    cov_trace = np.zeros(n_steps)
    imu_pos = flight_states[0, :3].copy()

    for i in range(n_steps):
        s = flight_states[i]
        true_xyz[i] = s[:3]

        accel_noisy = (
            s[6:9] + rng.normal(0, IMU_ACCEL_STD, 3)
            if len(s) >= 9
            else rng.normal(0, IMU_ACCEL_STD, 3)
        )
        if i > 0:
            imu_pos = imu_pos + accel_noisy * dt
        imu_xyz[i] = imu_pos

        ekf.predict(accel_noisy, dt)

        if i % gps_period == 0:
            gps_meas = s[:3] + rng.normal(0, GPS_STD, 3)
            ekf.update(gps_meas)
            gps_xyz_list.append(gps_meas.copy())

        ekf_xyz[i] = ekf.x[:3]
        cov_trace[i] = np.trace(ekf.P[:3, :3])

    gps_xyz = np.array(gps_xyz_list) if gps_xyz_list else np.zeros((1, 3))
    times = np.arange(n_steps) * dt
    err_ekf = np.sqrt(np.sum((true_xyz - ekf_xyz) ** 2, axis=1))
    err_imu = np.sqrt(np.sum((true_xyz - imu_xyz) ** 2, axis=1))

    # ── 3-Panel + inset ────────────────────────────────────────────────
    viz = ThreePanelViz(title="GPS + IMU Fusion (EKF)", world_size=WORLD_SIZE, figsize=(18, 9))
    viz.draw_buildings(world.obstacles)

    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    anim = SimAnimator("gps_imu_fusion", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_true = viz.create_trail_artists(color="black")
    trail_ekf = viz.create_trail_artists(color="dodgerblue")

    # GPS scatter (static, all at once)
    viz.ax3d.scatter(
        gps_xyz[:, 0],
        gps_xyz[:, 1],
        gps_xyz[:, 2],
        c="lime",
        s=4,
        alpha=0.25,
        label="GPS",
        zorder=3,
    )
    viz.ax_top.scatter(gps_xyz[:, 0], gps_xyz[:, 1], c="lime", s=3, alpha=0.25, zorder=3)
    viz.ax_side.scatter(gps_xyz[:, 0], gps_xyz[:, 2], c="lime", s=3, alpha=0.25, zorder=3)
    viz.ax3d.legend(fontsize=6, loc="upper left")

    # Inset for error + covariance
    ax_err = viz.fig.add_axes([0.62, 0.05, 0.34, 0.18])
    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(0, max(1.0, max(err_ekf.max(), err_imu.max()) * 1.1))
    ax_err.set_xlabel("Time [s]", fontsize=7)
    ax_err.set_ylabel("Pos. Error [m]", fontsize=7)
    ax_err.tick_params(labelsize=6)
    ax_err.grid(True, alpha=0.2)
    (err_ekf_line,) = ax_err.plot([], [], "dodgerblue", lw=0.8, label="EKF")
    (err_imu_line,) = ax_err.plot([], [], "r--", lw=0.6, alpha=0.5, label="IMU")
    ax_err.legend(fontsize=5, ncol=2)

    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail_true, true_xyz, k)
        viz.update_trail(trail_ekf, ekf_xyz, k)
        euler = flight_states[k, 3:6]
        viz.update_vehicle(true_xyz[k], euler, size=1.5)

        err_ekf_line.set_data(times[:k], err_ekf[:k])
        err_imu_line.set_data(times[:k], err_imu[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
