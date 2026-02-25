# Erwin Lejeune - 2026-02-19
"""UKF localisation: quadrotor flying with GPS sensor fusion.

The drone takes off, then flies a slow circular orbit.  True path vs.
UKF estimate vs. raw GPS measurements are overlaid.  The data panel
shows position error and covariance trace over time.

Reference: E. A. Wan, R. Van Der Merwe, "The Unscented Kalman Filter for
Nonlinear Estimation," AS-SPCC, 2000.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.estimation.ukf import UnscentedKalmanFilter
from uav_sim.logging import SimLogger
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.sensors.gps import GPS
from uav_sim.simulations.common import CRUISE_ALT, figure_8_path
from uav_sim.simulations.mission_runner import MissionResult, run_standard_mission
from uav_sim.simulations.standards import (
    SimulationStandard,
    deterministic_truth_trajectory,
    evaluate_completion,
)
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
BENCHMARK_MODE = True


def _truth_states(
    benchmark_mode: bool,
    world_obstacles: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MissionResult | None]:
    standard = (
        SimulationStandard.estimation_benchmark()
        if benchmark_mode
        else SimulationStandard.flight_coupled()
    )
    if benchmark_mode:
        states, times = deterministic_truth_trajectory(standard, alt=CRUISE_ALT, rx=8.0, ry=6.0)
        path = figure_8_path(
            duration=standard.duration,
            dt=0.1,
            alt=CRUISE_ALT,
            alt_amp=0.0,
            rx=8.0,
            ry=6.0,
        )
        return states, times, path, None
    path = figure_8_path(
        duration=standard.duration,
        dt=0.1,
        alt=CRUISE_ALT,
        alt_amp=0.0,
        rx=8.0,
        ry=6.0,
    )
    quad = Quadrotor()
    quad.reset(position=np.array([path[0, 0], path[0, 1], 0.0]))
    mission = run_standard_mission(
        quad,
        CascadedPIDController(),
        path,
        standard=standard,
        obstacles=world_obstacles,
    )
    times = np.arange(len(mission.states)) * standard.dt
    return mission.states, times, mission.tracking_path, mission


def main() -> None:
    world, _ = default_world()
    flight_states, times, path_3d, mission = _truth_states(BENCHMARK_MODE, world.obstacles)
    n_steps = len(flight_states)
    dt = (
        float(times[1] - times[0])
        if len(times) > 1
        else SimulationStandard.estimation_benchmark().dt
    )
    gps = GPS(noise_std=0.5, seed=42)

    def _f(x: np.ndarray, _u: np.ndarray, dt_: float) -> np.ndarray:
        return np.array(
            [
                x[0] + x[3] * dt_,
                x[1] + x[4] * dt_,
                x[2] + x[5] * dt_,
                x[3],
                x[4],
                x[5],
            ]
        )

    def _h(x: np.ndarray) -> np.ndarray:
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

    gps_period = max(1, int(1.0 / (5 * dt)))
    for i in range(n_steps):
        s = flight_states[i]
        true_xyz[i] = s[:3]
        gps_meas = gps.sense(s)
        meas_xyz[i] = gps_meas

        ukf.predict(np.zeros(3), dt)

        if i % gps_period == 0:
            ukf.update(gps_meas)

        est_xyz[i] = ukf.x[:3]
        cov_trace[i] = np.trace(ukf.P[:3, :3])

    err = np.sqrt(np.sum((true_xyz - est_xyz) ** 2, axis=1))
    completion = (
        mission.completion
        if mission is not None
        else evaluate_completion(
            true_xyz,
            path_3d[-1],
            dt=dt,
            standard=SimulationStandard.estimation_benchmark(),
            completed_tracking=True,
        )
    )

    logger = SimLogger("ukf", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "UKF")
    logger.log_metadata("benchmark_mode", BENCHMARK_MODE)
    logger.log_metadata("flight_coupled", not BENCHMARK_MODE)
    logger.log_metadata("dt", dt)
    logger.log_metadata("n_steps", n_steps)
    if mission is not None:
        logger.log_metadata("tracking_fallback", mission.tracking_fallback)
        logger.log_metadata("tracking_fallback_reason", mission.fallback_reason)
        logger.log_metadata("path_min_clearance_m", mission.path_min_clearance_m)
    for i in range(n_steps):
        logger.log_step(
            t=times[i],
            position=true_xyz[i],
            estimate=est_xyz[i],
            error=err[i],
            cov_trace=cov_trace[i],
        )
    logger.log_completion(**completion.as_dict())
    logger.log_summary("mean_error_m", float(err.mean()))
    logger.log_summary("max_error_m", float(err.max()))
    logger.save()

    # ── Visualisation ────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="UKF Localisation — GPS Fusion",
        world_size=WORLD_SIZE,
        figsize=(18, 9),
    )
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    # Data panel
    ax_d = viz.setup_data_axes(
        xlabel="Time [s]",
        ylabel="Error [m]",
        title="UKF Error & Covariance",
    )
    ax_d.set_xlim(0, times[-1])
    ax_d.set_ylim(0, max(1.0, err.max() * 1.2))
    (err_line,) = ax_d.plot([], [], "r-", lw=0.8, label="Pos err")
    (cov_line,) = ax_d.plot([], [], "b--", lw=0.6, label="tr(P)")
    ax_d.legend(fontsize=7)

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
