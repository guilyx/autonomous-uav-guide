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
from uav_sim.path_tracking.flight_ops import fly_path, init_hover, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.simulations.common import figure_8_path
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0
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
        est = cf.update(gyro, accel, DT)
        est_rp[i] = est[:2]

    times = np.arange(n_steps) * DT
    pos = flight_states[:, :3]

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
