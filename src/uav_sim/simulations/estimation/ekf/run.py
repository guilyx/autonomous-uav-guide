# Erwin Lejeune - 2026-02-21
"""EKF localisation: quadrotor flying a figure-8 with GPS fusion.

Three panels (3D, top-down, side) showing the quadrotor, with inset plots
for position error and covariance trace.  GPS measurements shown as scattered
dots, EKF estimate as a blue trail, true path in black.

Reference: S. Thrun et al., "Probabilistic Robotics," MIT Press, 2005, Ch. 3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.estimation.ekf import ExtendedKalmanFilter
from uav_sim.logging import SimLogger
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.sensors.gps import GPS
from uav_sim.simulations.common import figure_8_path
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def main() -> None:
    world, buildings = default_world()

    path_3d = figure_8_path(duration=60.0, dt=0.15, alt=CRUISE_ALT, alt_amp=0.0, rx=8.0, ry=6.0)

    quad = Quadrotor()
    quad.reset(position=np.array([path_3d[0, 0], path_3d[0, 1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=4.0, waypoint_threshold=1.0, adaptive=True)
    states_list: list[np.ndarray] = []
    from uav_sim.path_tracking.flight_ops import init_hover, takeoff

    init_hover(quad)
    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=0.005, duration=5.0, states=states_list)
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=180.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    n_steps = len(flight_states)
    dt = 0.005

    gps = GPS(noise_std=0.4, seed=42)

    def _f(x, _u, dt_):
        return np.array(
            [x[0] + x[3] * dt_, x[1] + x[4] * dt_, x[2] + x[5] * dt_, x[3], x[4], x[5]]
        )

    def _F(_x, _u, dt_):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt_
        return F

    def _h(x):
        return x[:3]

    def _H(_x):
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        return H

    ekf = ExtendedKalmanFilter(state_dim=6, meas_dim=3, f=_f, h=_h, F_jac=_F, H_jac=_H)
    ekf.Q = np.diag([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    ekf.R = np.diag([0.16, 0.16, 0.16])
    ekf.x = np.zeros(6)
    ekf.x[:3] = flight_states[0, :3]
    ekf.P = np.eye(6) * 0.5

    true_xyz = np.zeros((n_steps, 3))
    est_xyz = np.zeros((n_steps, 3))
    meas_xyz = np.zeros((n_steps, 3))
    cov_history = np.zeros((n_steps, 3, 3))

    for i in range(n_steps):
        s = flight_states[i]
        true_xyz[i] = s[:3]

        gps_meas = gps.sense(s)
        meas_xyz[i] = gps_meas

        ekf.predict(np.zeros(3), dt)
        ekf.update(gps_meas)
        est_xyz[i] = ekf.x[:3]
        cov_history[i] = ekf.P[:3, :3]

    times = np.arange(n_steps) * dt
    err = np.sqrt(np.sum((true_xyz - est_xyz) ** 2, axis=1))
    cov_trace = np.array([np.trace(cov_history[i]) for i in range(n_steps)])

    logger = SimLogger("ekf", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "EKF")
    logger.log_metadata("dt", dt)
    logger.log_metadata("n_steps", n_steps)
    for i in range(n_steps):
        logger.log_step(
            t=times[i],
            position=true_xyz[i],
            estimate=est_xyz[i],
            error=err[i],
            cov_trace=cov_trace[i],
        )
    logger.log_summary("mean_error_m", float(err.mean()))
    logger.log_summary("max_error_m", float(err.max()))
    logger.save()

    # ── 3-Panel viz ────────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="EKF Localisation — GPS Fusion", world_size=WORLD_SIZE, figsize=(18, 9)
    )
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    anim = SimAnimator("ekf", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_true = viz.create_trail_artists(color="black")
    trail_est = viz.create_trail_artists(color="dodgerblue")

    # GPS scatter
    skip_gps = max(1, n_steps // 300)
    gps_show = meas_xyz[::skip_gps]
    viz.ax3d.scatter(
        gps_show[:, 0],
        gps_show[:, 1],
        gps_show[:, 2],
        c="lime",
        s=4,
        alpha=0.2,
        label="GPS",
        zorder=3,
    )
    viz.ax_top.scatter(gps_show[:, 0], gps_show[:, 1], c="lime", s=3, alpha=0.2, zorder=3)
    viz.ax3d.legend(fontsize=6, loc="upper left")

    # Data panel: error + covariance
    ax_err = viz.setup_data_axes(
        xlabel="Time [s]",
        ylabel="Error [m]",
        title="EKF Error & Covariance",
    )
    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(0, max(1.0, err.max() * 1.2))
    (err_line,) = ax_err.plot([], [], "r-", lw=0.8, label="Pos err")
    (cov_line,) = ax_err.plot([], [], "b--", lw=0.6, label="tr(P)")
    ax_err.legend(fontsize=7)

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail_true, true_xyz, k)
        viz.update_trail(trail_est, est_xyz, k)
        viz.update_vehicle(true_xyz[k], flight_states[k, 3:6], size=1.5)
        err_line.set_data(times[:k], err[:k])
        cov_line.set_data(times[:k], cov_trace[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
