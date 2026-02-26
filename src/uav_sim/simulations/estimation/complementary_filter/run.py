# Erwin Lejeune - 2026-02-19
"""Complementary filter: attitude estimation on a flying quadrotor.

The drone takes off and flies a slow circular orbit.  The data panel
shows roll/pitch: true vs. estimated, making the fusion clearly visible.

Reference: R. Mahony et al., "Nonlinear Complementary Filters on the Special
Orthogonal Group," IEEE TAC, 2008.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.estimation.complementary_filter import ComplementaryFilter
from uav_sim.logging import SimLogger
from uav_sim.path_tracking.pid_controller import CascadedPIDController
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
    rng = np.random.default_rng(42)
    world, _ = default_world()
    flight_states, times, path_3d, mission = _truth_states(BENCHMARK_MODE, world.obstacles)
    n_steps = len(flight_states)
    dt = (
        float(times[1] - times[0])
        if len(times) > 1
        else SimulationStandard.estimation_benchmark().dt
    )

    cf = ComplementaryFilter(alpha=0.98)
    g = 9.81

    true_rp = np.zeros((n_steps, 2))
    est_rp = np.zeros((n_steps, 2))

    for i in range(n_steps):
        s = flight_states[i]
        roll_true, pitch_true = s[3], s[4]
        true_rp[i] = [roll_true, pitch_true]

        gyro = s[9:12] + rng.normal(0, 0.03, 3) if len(s) >= 12 else rng.normal(0, 0.03, 3)
        accel = np.array(
            [
                -g * np.sin(pitch_true) + rng.normal(0, 0.2),
                g * np.sin(roll_true) * np.cos(pitch_true) + rng.normal(0, 0.2),
                g * np.cos(roll_true) * np.cos(pitch_true) + rng.normal(0, 0.2),
            ]
        )
        est = cf.update(gyro, accel, dt)
        est_rp[i] = est[:2]

    pos = flight_states[:, :3]
    completion = (
        mission.completion
        if mission is not None
        else evaluate_completion(
            pos,
            path_3d[-1],
            dt=dt,
            standard=SimulationStandard.estimation_benchmark(),
            completed_tracking=True,
        )
    )

    roll_err = np.abs(true_rp[:, 0] - est_rp[:, 0])
    pitch_err = np.abs(true_rp[:, 1] - est_rp[:, 1])
    logger = SimLogger("complementary_filter", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "Complementary Filter")
    logger.log_metadata("benchmark_mode", BENCHMARK_MODE)
    logger.log_metadata("flight_coupled", not BENCHMARK_MODE)
    logger.log_metadata("dt", dt)
    logger.log_metadata("alpha", 0.98)
    if mission is not None:
        logger.log_metadata("tracking_fallback", mission.tracking_fallback)
        logger.log_metadata("tracking_fallback_reason", mission.fallback_reason)
        logger.log_metadata("path_min_clearance_m", mission.path_min_clearance_m)
    for i in range(n_steps):
        logger.log_step(
            t=times[i],
            position=pos[i],
            true_roll=true_rp[i, 0],
            true_pitch=true_rp[i, 1],
            est_roll=est_rp[i, 0],
            est_pitch=est_rp[i, 1],
            roll_error=roll_err[i],
            pitch_error=pitch_err[i],
        )
    logger.log_completion(**completion.as_dict())
    logger.log_summary("mean_roll_error_rad", float(roll_err.mean()))
    logger.log_summary("mean_pitch_error_rad", float(pitch_err.mean()))
    logger.save()

    # ── Visualisation ────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="Complementary Filter — Attitude Estimation",
        world_size=WORLD_SIZE,
        figsize=(18, 9),
    )
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    # Data panel: roll/pitch comparison
    ax_d = viz.setup_data_axes(
        xlabel="Time [s]",
        ylabel="Angle [rad]",
        title="Attitude: True vs CF",
    )
    ax_d.set_xlim(0, times[-1])
    ylim = max(0.15, np.abs(true_rp).max() * 1.3)
    ax_d.set_ylim(-ylim, ylim)
    (l_tr,) = ax_d.plot([], [], "k-", lw=0.8, label="Roll true")
    (l_er,) = ax_d.plot([], [], "b-", lw=0.7, label="Roll CF")
    (l_tp,) = ax_d.plot([], [], "k--", lw=0.8, label="Pitch true")
    (l_ep,) = ax_d.plot([], [], "r--", lw=0.7, label="Pitch CF")
    ax_d.legend(fontsize=6, ncol=2, loc="upper right")

    anim = SimAnimator("complementary_filter", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="orange")

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail_arts, pos, k)
        viz.update_vehicle(pos[k], flight_states[k, 3:6], size=1.5)
        l_tr.set_data(times[:k], true_rp[:k, 0])
        l_er.set_data(times[:k], est_rp[:k, 0])
        l_tp.set_data(times[:k], true_rp[:k, 1])
        l_ep.set_data(times[:k], est_rp[:k, 1])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
