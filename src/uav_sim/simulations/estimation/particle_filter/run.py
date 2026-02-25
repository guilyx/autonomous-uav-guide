# Erwin Lejeune - 2026-02-19
"""Particle filter localisation: quadrotor with GPS in 30 m urban world.

The drone takes off and flies a slow circular orbit.  The particle cloud
is rendered in the top-down view.  The data panel shows N_eff and position
error over time.

Reference: M. S. Arulampalam et al., "A Tutorial on Particle Filters," IEEE
TSP, 2002.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.estimation.particle_filter import ParticleFilter
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
GPS_STD = 0.6
N_PARTICLES = 200
BENCHMARK_MODE = True


def _f_single(
    x: np.ndarray,
    _u: np.ndarray,
    dt_: float,
) -> np.ndarray:
    return np.array([x[0] + x[2] * dt_, x[1] + x[3] * dt_, x[2], x[3]])


def _likelihood(z: np.ndarray, x: np.ndarray) -> float:
    diff = z[:2] - x[:2]
    return float(np.exp(-0.5 * np.dot(diff, diff) / GPS_STD**2))


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

    pf_dt = 0.05
    dt_flight = (
        float(times[1] - times[0])
        if len(times) > 1
        else SimulationStandard.estimation_benchmark().dt
    )
    pf_skip = max(1, int(pf_dt / dt_flight))
    pf_states = flight_states[::pf_skip]
    n_steps = len(pf_states)

    pf = ParticleFilter(
        state_dim=4,
        num_particles=N_PARTICLES,
        f=_f_single,
        likelihood=_likelihood,
        process_noise_std=0.3,
    )
    init_state = np.array([pf_states[0, 0], pf_states[0, 1], 0.0, 0.0])
    pf.reset(init_state, spread=1.5)

    true_xy = np.zeros((n_steps, 2))
    est_xy = np.zeros((n_steps, 2))
    part_hist: list[np.ndarray] = []
    weight_hist: list[np.ndarray] = []
    n_eff_hist = np.zeros(n_steps)

    for i in range(n_steps):
        s = pf_states[i]
        true_xy[i] = s[:2]
        gps_meas = s[:2] + rng.normal(0, GPS_STD, 2)

        pf.predict(np.zeros(1), pf_dt)
        pf.update(gps_meas)

        est_xy[i] = pf.estimate[:2]
        part_hist.append(pf.particles[:, :2].copy())
        weight_hist.append(pf.weights.copy())
        n_eff_hist[i] = 1.0 / np.sum(pf.weights**2)

    times = np.arange(n_steps) * pf_dt
    err = np.sqrt(np.sum((true_xy - est_xy) ** 2, axis=1))
    pos_3d_true = np.column_stack([true_xy, np.full(n_steps, CRUISE_ALT)])
    pos_3d_est = np.column_stack([est_xy, np.full(n_steps, CRUISE_ALT)])
    completion = (
        mission.completion
        if mission is not None
        else evaluate_completion(
            pos_3d_true,
            path_3d[-1],
            dt=pf_dt,
            standard=SimulationStandard.estimation_benchmark(),
            completed_tracking=True,
        )
    )

    logger = SimLogger("particle_filter", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "Particle Filter")
    logger.log_metadata("benchmark_mode", BENCHMARK_MODE)
    logger.log_metadata("flight_coupled", not BENCHMARK_MODE)
    logger.log_metadata("dt", pf_dt)
    logger.log_metadata("n_particles", N_PARTICLES)
    if mission is not None:
        logger.log_metadata("tracking_fallback", mission.tracking_fallback)
        logger.log_metadata("tracking_fallback_reason", mission.fallback_reason)
        logger.log_metadata("path_min_clearance_m", mission.path_min_clearance_m)
    for i in range(n_steps):
        logger.log_step(
            t=times[i], position=true_xy[i], estimate=est_xy[i], error=err[i], n_eff=n_eff_hist[i]
        )
    logger.log_completion(**completion.as_dict())
    logger.log_summary("mean_error_m", float(err.mean()))
    logger.log_summary("max_error_m", float(err.max()))
    logger.save()

    # ── Visualisation ────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="Particle Filter Localisation",
        world_size=WORLD_SIZE,
        figsize=(18, 9),
    )
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    # Data panel: error + N_eff
    ax_d = viz.setup_data_axes(
        xlabel="Time [s]",
        ylabel="Pos. Error [m]",
        title="PF Error & N_eff",
    )
    ax_d.set_xlim(0, times[-1])
    ax_d.set_ylim(0, max(2.0, err.max() * 1.2))
    (err_line,) = ax_d.plot([], [], "r-", lw=0.8, label="Error")
    ax_d.legend(fontsize=7, loc="upper left")

    ax_neff = ax_d.twinx()
    ax_neff.set_ylim(0, N_PARTICLES * 1.2)
    ax_neff.set_ylabel("N_eff", fontsize=8, color="blue")
    ax_neff.tick_params(labelsize=7, colors="blue")
    (neff_line,) = ax_neff.plot([], [], "b--", lw=0.6)

    anim = SimAnimator("particle_filter", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_true = viz.create_trail_artists(color="black")
    trail_est = viz.create_trail_artists(color="dodgerblue")

    cloud = viz.ax_top.scatter(
        [],
        [],
        s=3,
        c=[],
        cmap="cool",
        alpha=0.4,
        vmin=0,
        vmax=1,
        zorder=4,
    )

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    pf_to_flight = [min(i * pf_skip, len(flight_states) - 1) for i in range(n_steps)]

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail_true, pos_3d_true, k)
        viz.update_trail(trail_est, pos_3d_est, k)
        fi = pf_to_flight[k]
        viz.update_vehicle(
            np.array([est_xy[k, 0], est_xy[k, 1], CRUISE_ALT]),
            flight_states[fi, 3:6],
            size=1.5,
        )

        pts = part_hist[k]
        w = weight_hist[k]
        w_norm = w / (w.max() + 1e-12)
        cloud.set_offsets(pts)
        cloud.set_array(w_norm)

        err_line.set_data(times[:k], err[:k])
        neff_line.set_data(times[:k], n_eff_hist[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
