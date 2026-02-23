# Erwin Lejeune - 2026-02-19
"""GPS + IMU sensor fusion via EKF for quadrotor localisation.

The drone takes off, then flies a slow circular orbit at constant
velocity so estimation results are clearly visible. Three estimates
are compared: IMU-only dead-reckoning, raw GPS fixes, and fused EKF.

Reference: S. Thrun et al., "Probabilistic Robotics," MIT Press, 2005, Ch. 3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.estimation.ekf import ExtendedKalmanFilter
from uav_sim.logging import SimLogger
from uav_sim.path_tracking.flight_ops import fly_path, init_hover, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.simulations.common import figure_8_path
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0
GPS_STD = 0.5
IMU_ACCEL_STD = 0.10
GPS_RATE_HZ = 5
DT = 0.005


def main() -> None:
    rng = np.random.default_rng(42)
    world, buildings = default_world()

    path_3d = figure_8_path(duration=45.0, dt=0.15, alt=CRUISE_ALT, alt_amp=0.0, rx=8.0, ry=6.0)

    quad = Quadrotor()
    quad.reset(position=np.array([path_3d[0, 0], path_3d[0, 1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=4.0, waypoint_threshold=2.0, adaptive=True)

    states_list: list[np.ndarray] = []
    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=DT, duration=3.0, states=states_list)
    init_hover(quad)
    fly_path(quad, ctrl, path_3d, dt=DT, pursuit=pursuit, timeout=180.0, states=states_list)

    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    n_steps = len(flight_states)

    # ── EKF setup: state = [x, y, z, vx, vy, vz] ──────────────────────
    def _f(x: np.ndarray, u: np.ndarray, dt_: float) -> np.ndarray:
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

    def _F(_x: np.ndarray, _u: np.ndarray, dt_: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = dt_
        F[1, 4] = dt_
        F[2, 5] = dt_
        return F

    def _h(x: np.ndarray) -> np.ndarray:
        return x[:3]

    def _H(_x: np.ndarray) -> np.ndarray:
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        return H

    ekf = ExtendedKalmanFilter(state_dim=6, meas_dim=3, f=_f, h=_h, F_jac=_F, H_jac=_H)
    ekf.Q = np.diag(
        [
            0.001,
            0.001,
            0.001,
            IMU_ACCEL_STD**2,
            IMU_ACCEL_STD**2,
            IMU_ACCEL_STD**2,
        ]
    )
    ekf.R = np.diag([GPS_STD**2, GPS_STD**2, GPS_STD**2])
    ekf.x = np.zeros(6)
    ekf.x[:3] = flight_states[0, :3]
    ekf.P = np.eye(6) * 0.5

    gps_period = int(1.0 / (GPS_RATE_HZ * DT))

    true_xyz = np.zeros((n_steps, 3))
    ekf_xyz = np.zeros((n_steps, 3))
    imu_xyz = np.zeros((n_steps, 3))
    gps_xyz_list: list[np.ndarray] = []
    cov_trace = np.zeros(n_steps)
    imu_pos = flight_states[0, :3].copy()

    for i in range(n_steps):
        s = flight_states[i]
        true_xyz[i] = s[:3]

        accel_noisy = s[6:9] + rng.normal(0, IMU_ACCEL_STD, 3)
        if i > 0:
            imu_pos = imu_pos + accel_noisy * DT
        imu_xyz[i] = imu_pos

        ekf.predict(accel_noisy, DT)

        if i % gps_period == 0:
            gps_meas = s[:3] + rng.normal(0, GPS_STD, 3)
            ekf.update(gps_meas)
            gps_xyz_list.append(gps_meas.copy())

        ekf_xyz[i] = ekf.x[:3]
        cov_trace[i] = np.trace(ekf.P[:3, :3])

    gps_xyz = np.array(gps_xyz_list) if gps_xyz_list else np.zeros((1, 3))
    times = np.arange(n_steps) * DT
    err_ekf = np.sqrt(np.sum((true_xyz - ekf_xyz) ** 2, axis=1))
    err_imu = np.sqrt(np.sum((true_xyz - imu_xyz) ** 2, axis=1))

    logger = SimLogger("gps_imu_fusion", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "GPS+IMU EKF Fusion")
    logger.log_metadata("dt", DT)
    logger.log_metadata("gps_rate_hz", GPS_RATE_HZ)
    for i in range(n_steps):
        logger.log_step(
            t=times[i],
            position=true_xyz[i],
            ekf_estimate=ekf_xyz[i],
            imu_estimate=imu_xyz[i],
            ekf_error=err_ekf[i],
            imu_error=err_imu[i],
        )
    logger.log_summary("mean_ekf_error_m", float(err_ekf.mean()))
    logger.log_summary("mean_imu_error_m", float(err_imu.mean()))
    logger.save()

    # ── Visualisation: 3D + top-down + data panel ─────────────────────
    viz = ThreePanelViz(title="GPS + IMU Fusion (EKF)", world_size=WORLD_SIZE, figsize=(18, 9))
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    # Data panel: position error
    ax_d = viz.setup_data_axes(xlabel="Time [s]", ylabel="Position Error [m]", title="EKF vs IMU")
    ax_d.set_xlim(0, times[-1])
    ax_d.set_ylim(0, max(1.0, max(err_ekf.max(), err_imu.max()) * 1.1))
    (err_ekf_line,) = ax_d.plot([], [], "dodgerblue", lw=0.8, label="EKF")
    (err_imu_line,) = ax_d.plot([], [], "r--", lw=0.6, alpha=0.5, label="IMU-only")
    ax_d.legend(fontsize=7, loc="upper left")

    trail_true = viz.create_trail_artists(color="black")
    trail_ekf = viz.create_trail_artists(color="dodgerblue")

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
    viz.ax3d.legend(fontsize=6, loc="upper left")

    anim = SimAnimator("gps_imu_fusion", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail_true, true_xyz, k)
        viz.update_trail(trail_ekf, ekf_xyz, k)
        viz.update_vehicle(true_xyz[k], flight_states[k, 3:6], size=1.5)

        err_ekf_line.set_data(times[:k], err_ekf[:k])
        err_imu_line.set_data(times[:k], err_imu[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
