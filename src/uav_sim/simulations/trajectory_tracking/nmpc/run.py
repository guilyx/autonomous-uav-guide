# Erwin Lejeune - 2026-02-19
"""NMPC online trajectory tracking — figure-8.

A nonlinear MPC (single-shooting, RK2 integration) runs as a
receding-horizon controller at 50 Hz, tracking a figure-8 reference
trajectory.

Reference: M. Diehl et al., "Real-Time Optimization and Nonlinear Model
Predictive Control of Processes Governed by Differential-Algebraic
Equations," J. Process Control, 2002. DOI: 10.1016/S0959-1524(02)00023-1
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import init_hover
from uav_sim.simulations.common import (
    WORLD_SIZE,
    figure_8_ref,
    frame_indices,
)
from uav_sim.trajectory_tracking.nmpc import NMPCTracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

DT_SIM = 0.01
DT_CTRL = 0.02


def main() -> None:
    world, buildings = default_world()

    quad = Quadrotor()
    rp0, _ = figure_8_ref(0.0)
    quad.reset(position=rp0.copy())
    init_hover(quad)

    nmpc = NMPCTracker(
        horizon=6,
        dt=DT_CTRL,
        mass=quad.params.mass,
        gravity=quad.params.gravity,
        inertia=quad.params.inertia,
    )

    dur = 15.0
    sim_steps_per_ctrl = max(1, int(DT_CTRL / DT_SIM))
    max_ctrl_steps = int(dur / DT_CTRL)

    states_list: list[np.ndarray] = []
    refs_list: list[np.ndarray] = []
    wrench = quad.hover_wrench()

    for ci in range(max_ctrl_steps):
        s = quad.state
        if not (np.all(np.isfinite(s[:3])) and np.all(np.abs(s[:3]) < 500)):
            break

        t = ci * DT_CTRL
        rp, rv = figure_8_ref(t)
        wrench = nmpc.compute(s, rp, ref_vel=rv)

        for _ in range(sim_steps_per_ctrl):
            states_list.append(quad.state.copy())
            refs_list.append(rp.copy())
            quad.step(wrench, DT_SIM)

    states = np.array(states_list) if states_list else np.zeros((1, 12))
    refs = np.array(refs_list) if refs_list else np.zeros((1, 3))
    pos = states[:, :3]
    n_total = len(pos)
    times = np.arange(n_total) * DT_SIM
    err = np.linalg.norm(pos - refs, axis=1)
    speed = np.linalg.norm(states[:, 6:9], axis=1)

    # ── Animation ─────────────────────────────────────────────────────
    idx = frame_indices(n_total, max_frames=60)
    n_frames = len(idx)

    viz = ThreePanelViz(title="NMPC — Figure-8 Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(refs, color="red", lw=1.0, alpha=0.3, label="Reference")

    trail = viz.create_trail_artists(color="orange")
    (ref_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, zorder=10)
    (ref_top,) = viz.ax_top.plot([], [], "r*", ms=8, zorder=10)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    ax_d = viz.setup_data_axes(title="Tracking Error [m]", ylabel="Error")
    ax_d.set_xlim(0, dur)
    ax_d.set_ylim(0, max(0.5, err.max() * 1.1))
    (l_err,) = ax_d.plot([], [], "r-", lw=0.8, label="||e||")
    ax_v = ax_d.twinx()
    ax_v.set_ylabel("Speed [m/s]", fontsize=7)
    ax_v.tick_params(labelsize=6)
    ax_v.set_ylim(0, max(1.0, speed.max() * 1.2))
    (l_spd,) = ax_v.plot([], [], "b-", lw=0.5, alpha=0.6, label="speed")
    ax_d.legend(fontsize=5, loc="upper right")
    ax_v.legend(fontsize=5, loc="lower right")

    anim = SimAnimator("nmpc", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig
    title = viz.ax3d.set_title("NMPC")

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        ref_3d.set_data([refs[k, 0]], [refs[k, 1]])
        ref_3d.set_3d_properties([refs[k, 2]])
        ref_top.set_data([refs[k, 0]], [refs[k, 1]])
        l_err.set_data(times[:k], err[:k])
        l_spd.set_data(times[:k], speed[:k])
        title.set_text(f"NMPC — t={times[k]:.1f}s  err={err[k]:.2f}m")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
