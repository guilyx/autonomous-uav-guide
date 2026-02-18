# Erwin Lejeune - 2026-02-18
"""Particle filter localisation: quadrotor with GPS in 30 m urban world.

Three panels (3D, top-down, side) showing the quadrotor flying a circle.
The particle cloud is rendered in the top-down panel as colour-coded dots
(weight = colour intensity).  Inset panels show N_eff and position error.

Reference: M. S. Arulampalam et al., "A Tutorial on Particle Filters," IEEE
TSP, 2002.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.estimation.particle_filter import ParticleFilter
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0
GPS_STD = 0.6
N_PARTICLES = 200


def _f_single(x, _u, dt_):
    return np.array([x[0] + x[2] * dt_, x[1] + x[3] * dt_, x[2], x[3]])


def _likelihood(z, x):
    diff = z[:2] - x[:2]
    return float(np.exp(-0.5 * np.dot(diff, diff) / GPS_STD**2))


def main() -> None:
    rng = np.random.default_rng(42)

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

    # Downsample for PF (PF at 20 Hz)
    pf_dt = 0.05
    pf_skip = max(1, int(pf_dt / 0.005))
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

    # ── 3-Panel viz ────────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="Particle Filter Localisation", world_size=WORLD_SIZE, figsize=(18, 9)
    )
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    anim = SimAnimator("particle_filter", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_true = viz.create_trail_artists(color="black")
    trail_est = viz.create_trail_artists(color="dodgerblue")

    # Particle cloud in top-down view (dynamic)
    cloud = viz.ax_top.scatter([], [], s=3, c=[], cmap="cool", alpha=0.4, vmin=0, vmax=1, zorder=4)
    # Inset: error + N_eff
    ax_inset = viz.fig.add_axes([0.58, 0.03, 0.38, 0.22])
    ax_inset.set_xlim(0, times[-1])
    ax_inset.set_ylim(0, max(2.0, err.max() * 1.2))
    ax_inset.set_xlabel("Time [s]", fontsize=7)
    ax_inset.set_ylabel("Pos. Error [m]", fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.grid(True, alpha=0.2)
    (err_line,) = ax_inset.plot([], [], "r-", lw=0.8, label="Error")
    ax_inset.legend(fontsize=5)

    ax_neff = ax_inset.twinx()
    ax_neff.set_ylim(0, N_PARTICLES * 1.2)
    ax_neff.set_ylabel("N_eff", fontsize=7, color="blue")
    ax_neff.tick_params(labelsize=6, colors="blue")
    (neff_line,) = ax_neff.plot([], [], "b--", lw=0.6, label="N_eff")

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    # Map PF indices to flight_states indices for euler angles
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

        # Particle cloud update
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
