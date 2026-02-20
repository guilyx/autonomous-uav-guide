# Erwin Lejeune - 2026-02-18
"""UKF localisation: quadrotor flying with GPS sensor fusion.

Three panels (3D, top-down, side) in a 30 m urban environment.  True path vs.
UKF estimate vs. raw GPS measurements are overlaid, with inset plots for
position error, innovation norm, and covariance trace.

Reference: E. A. Wan, R. Van Der Merwe, "The Unscented Kalman Filter for
Nonlinear Estimation," AS-SPCC, 2000.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.estimation.ukf import UnscentedKalmanFilter
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.sensors.gps import GPS
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def main() -> None:
    world, buildings = default_world()

    cx, cy, radius = 15.0, 15.0, 8.0
    n_wp = 60
    angles = np.linspace(0, 1.8 * np.pi, n_wp)
    path_3d = np.column_stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles), np.full(n_wp, CRUISE_ALT)]
    )

    quad = Quadrotor()
    quad.reset(position=np.array([cx + radius, cy, CRUISE_ALT]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=60.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    n_steps = len(flight_states)
    dt = 0.005
    gps = GPS(noise_std=0.5, seed=42)

    def _f(x, _u, dt_):
        return np.array(
            [x[0] + x[3] * dt_, x[1] + x[4] * dt_, x[2] + x[5] * dt_, x[3], x[4], x[5]]
        )

    def _h(x):
        return x[:3]

    ukf = UnscentedKalmanFilter(state_dim=6, meas_dim=3, f=_f, h=_h)
    ukf.Q = np.diag([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    ukf.R = np.diag([0.25, 0.25, 0.25])
    ukf.x = np.zeros(6)
    ukf.x[:3] = flight_states[0, :3]
    ukf.P = np.eye(6) * 0.5

    true_xyz = np.zeros((n_steps, 3))
    est_xyz = np.zeros((n_steps, 3))
    meas_xyz = np.zeros((n_steps, 3))
    cov_trace = np.zeros(n_steps)
    innovation_norm = np.zeros(n_steps)

    gps_period = max(1, int(1.0 / (5 * dt)))
    for i in range(n_steps):
        s = flight_states[i]
        true_xyz[i] = s[:3]
        gps_meas = gps.sense(s)
        meas_xyz[i] = gps_meas

        ukf.predict(np.zeros(3), dt)
        inn = gps_meas - _h(ukf.x)
        innovation_norm[i] = float(np.linalg.norm(inn))

        if i % gps_period == 0:
            ukf.update(gps_meas)

        est_xyz[i] = ukf.x[:3]
        cov_trace[i] = np.trace(ukf.P[:3, :3])

    times = np.arange(n_steps) * dt
    err = np.sqrt(np.sum((true_xyz - est_xyz) ** 2, axis=1))

    # ── 3-Panel viz ────────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="UKF Localisation — GPS Fusion", world_size=WORLD_SIZE, figsize=(18, 9)
    )
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    anim = SimAnimator("ukf", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_true = viz.create_trail_artists(color="black")
    trail_est = viz.create_trail_artists(color="dodgerblue")

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

    ax_err = viz.fig.add_axes([0.62, 0.05, 0.34, 0.18])
    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(0, max(1.0, err.max() * 1.2))
    ax_err.set_xlabel("Time [s]", fontsize=7)
    ax_err.set_ylabel("Error [m]", fontsize=7)
    ax_err.tick_params(labelsize=6)
    ax_err.grid(True, alpha=0.2)
    (err_line,) = ax_err.plot([], [], "r-", lw=0.8, label="Pos err")
    (cov_line,) = ax_err.plot([], [], "b--", lw=0.6, label="tr(P)")
    ax_err.legend(fontsize=5, ncol=2)

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
