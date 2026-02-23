# Erwin Lejeune - 2026-02-15
"""MPC path tracking: 3-panel live simulation.

The quadrotor tracks a figure-8 reference using a receding-horizon
linear MPC, rendered in an urban 30 m world.

Reference: J. B. Rawlings et al., "Model Predictive Control: Theory,
Computation, and Design," Nob Hill, 2017.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import init_hover
from uav_sim.path_tracking.mpc_controller import MPCController
from uav_sim.simulations.common import WORLD_SIZE, figure_8_ref, frame_indices
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")


def main() -> None:
    world, buildings = default_world()

    quad = Quadrotor()
    rp0, _ = figure_8_ref(0.0)
    quad.reset(position=rp0.copy())
    init_hover(quad)

    ctrl = MPCController(
        horizon=8,
        dt=0.05,
        mass=quad.params.mass,
        gravity=quad.params.gravity,
        inertia=quad.params.inertia,
    )

    dt, dur = 0.01, 20.0
    ctrl_dt = 0.05
    steps = int(dur / dt)
    states = np.zeros((steps, 12))
    refs = np.zeros((steps, 3))
    times = np.zeros(steps)
    wrench = quad.hover_wrench()

    ctrl_counter = 0
    for i in range(steps):
        t = i * dt
        rp, rv = figure_8_ref(t)
        refs[i] = rp
        states[i] = quad.state
        times[i] = t

        ctrl_counter += dt
        if ctrl_counter >= ctrl_dt - 1e-8:
            wrench = ctrl.compute(quad.state, rp, target_vel=rv)
            ctrl_counter = 0.0
        quad.step(wrench, dt)

    pos = states[:, :3]
    err = np.linalg.norm(pos - refs, axis=1)

    # ── Visualisation ─────────────────────────────────────────────────
    idx = frame_indices(steps, max_frames=120)
    n_frames = len(idx)

    viz = ThreePanelViz(title="MPC Path Tracking — Figure-8", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(refs, color="red", lw=1.0, alpha=0.3, label="Reference")

    trail = viz.create_trail_artists()
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
    speed = np.linalg.norm(states[:, 6:9], axis=1)
    ax_v.set_ylim(0, max(1.0, speed.max() * 1.2))
    (l_spd,) = ax_v.plot([], [], "b-", lw=0.5, alpha=0.6, label="speed")
    ax_d.legend(fontsize=5, loc="upper right")
    ax_v.legend(fontsize=5, loc="lower right")

    anim = SimAnimator("mpc_tracking", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig
    title = viz.ax3d.set_title("MPC Tracking")

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        ref_3d.set_data([refs[k, 0]], [refs[k, 1]])
        ref_3d.set_3d_properties([refs[k, 2]])
        ref_top.set_data([refs[k, 0]], [refs[k, 1]])
        l_err.set_data(times[:k], err[:k])
        l_spd.set_data(times[:k], speed[:k])
        title.set_text(f"MPC Tracking — t={times[k]:.1f}s  err={err[k]:.2f}m")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
