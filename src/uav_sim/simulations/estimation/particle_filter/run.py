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
from uav_sim.path_tracking.flight_ops import fly_path, init_hover, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.simulations.common import figure_8_path
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0
GPS_STD = 0.6
N_PARTICLES = 200
DT = 0.005


def _f_single(
    x: np.ndarray,
    _u: np.ndarray,
    dt_: float,
) -> np.ndarray:
    return np.array([x[0] + x[2] * dt_, x[1] + x[3] * dt_, x[2], x[3]])


def _likelihood(z: np.ndarray, x: np.ndarray) -> float:
    diff = z[:2] - x[:2]
    return float(np.exp(-0.5 * np.dot(diff, diff) / GPS_STD**2))


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

    pf_dt = 0.05
    pf_skip = max(1, int(pf_dt / DT))
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

    logger = SimLogger("particle_filter", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "Particle Filter")
    logger.log_metadata("dt", pf_dt)
    logger.log_metadata("n_particles", N_PARTICLES)
    for i in range(n_steps):
        logger.log_step(
            t=times[i], position=true_xy[i], estimate=est_xy[i], error=err[i], n_eff=n_eff_hist[i]
        )
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
            np.array([true_xy[k, 0], true_xy[k, 1], CRUISE_ALT]),
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
