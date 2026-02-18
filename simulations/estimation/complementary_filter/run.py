# Erwin Lejeune - 2026-02-18
"""Complementary filter: attitude estimation on a flying quadrotor.

Three panels (3D, top-down, side) showing the quadrotor flying a circle in a
30 m urban environment.  Inset panels display the roll/pitch fusion and the
error between true and estimated attitude.

The complementary filter fuses high-rate gyro integration with low-rate
accelerometer-derived angles.

Reference: R. Mahony et al., "Nonlinear Complementary Filters on the Special
Orthogonal Group," IEEE TAC, 2008.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.estimation.complementary_filter import ComplementaryFilter
from uav_sim.path_tracking.flight_ops import fly_path, init_hover
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def main() -> None:
    rng = np.random.default_rng(42)
    world = World(bounds_min=np.zeros(3), bounds_max=np.full(3, WORLD_SIZE))
    add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=5, seed=42)

    cx, cy, radius = 15.0, 15.0, 8.0
    n_wp = 60
    angles = np.linspace(0, 1.8 * np.pi, n_wp)
    path_3d = np.column_stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles), np.full(n_wp, CRUISE_ALT)]
    )

    quad = Quadrotor()
    quad.reset(position=np.array([cx + radius, cy, CRUISE_ALT]))
    init_hover(quad)
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=40.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    n_steps = len(flight_states)
    dt = 0.005

    cf = ComplementaryFilter(alpha=0.98)
    g = 9.81

    true_rp = np.zeros((n_steps, 2))
    est_rp = np.zeros((n_steps, 2))
    err_rp = np.zeros((n_steps, 2))

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
        err_rp[i] = np.abs(true_rp[i] - est_rp[i])

    times = np.arange(n_steps) * dt
    pos = flight_states[:, :3]

    # ── 3-Panel viz ────────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="Complementary Filter — Attitude Estimation", world_size=WORLD_SIZE, figsize=(18, 9)
    )
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.7, alpha=0.3, label="Plan")

    anim = SimAnimator("complementary_filter", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="orange")

    # Inset: roll + pitch comparison
    ax_att = viz.fig.add_axes([0.58, 0.03, 0.38, 0.22])
    ax_att.set_xlim(0, times[-1])
    ylim = max(0.15, np.abs(true_rp).max() * 1.3)
    ax_att.set_ylim(-ylim, ylim)
    ax_att.set_xlabel("Time [s]", fontsize=7)
    ax_att.set_ylabel("Angle [rad]", fontsize=7)
    ax_att.tick_params(labelsize=6)
    ax_att.grid(True, alpha=0.2)
    (l_tr,) = ax_att.plot([], [], "k-", lw=0.8, label="Roll true")
    (l_er,) = ax_att.plot([], [], "b-", lw=0.7, label="Roll CF")
    (l_tp,) = ax_att.plot([], [], "k--", lw=0.8, label="Pitch true")
    (l_ep,) = ax_att.plot([], [], "r--", lw=0.7, label="Pitch CF")
    ax_att.legend(fontsize=5, ncol=2, loc="upper right")

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
