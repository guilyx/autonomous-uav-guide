# Erwin Lejeune - 2026-02-15
"""Geometric SO(3) controller: 3-panel figure-8 tracking.

Demonstrates the geometric controller tracking a figure-8 reference
trajectory.  The controller works on SO(3) and avoids Euler-angle
singularities, producing smooth torque commands from the rotation
error between the current and desired orientation.

Reference: T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of
a Quadrotor UAV on SE(3)," CDC, 2010. DOI: 10.1109/CDC.2010.5717652
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import init_hover
from uav_sim.path_tracking.geometric_controller import GeometricController
from uav_sim.simulations.common import (
    STANDARD_DURATION,
    WORLD_SIZE,
    figure_8_ref,
    frame_indices,
)
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

    ctrl = GeometricController()

    dt, dur = 0.01, STANDARD_DURATION
    steps = int(dur / dt)
    states = np.zeros((steps, 12))
    refs = np.zeros((steps, 3))
    times = np.zeros(steps)
    thrust_hist = np.zeros(steps)

    for i in range(steps):
        t = i * dt
        rp, rv = figure_8_ref(t)
        refs[i] = rp
        states[i] = quad.state
        times[i] = t
        w = ctrl.compute(quad.state, rp, target_vel=rv)
        thrust_hist[i] = w[0]
        quad.step(w, dt)

    pos = states[:, :3]
    err = np.linalg.norm(pos - refs, axis=1)

    # ── visualisation ──────────────────────────────────────────────────
    idx = frame_indices(steps)
    n_frames = len(idx)

    viz = ThreePanelViz(title="Geometric SO(3) — Figure-8 Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(refs, color="red", lw=1.0, alpha=0.3, label="Reference")

    (ref_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, zorder=10)
    (ref_top,) = viz.ax_top.plot([], [], "r*", ms=8, zorder=10)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    trail = viz.create_trail_artists()

    ax_d = viz.setup_data_axes(title="Tracking Error [m]", ylabel="Error")
    ax_d.set_xlim(0, dur)
    ax_d.set_ylim(0, max(0.5, err.max() * 1.1))
    (l_err,) = ax_d.plot([], [], "r-", lw=0.8, label="||e||")
    ax_t = ax_d.twinx()
    ax_t.set_ylabel("Thrust [N]", fontsize=7)
    ax_t.tick_params(labelsize=6)
    ax_t.set_ylim(0, max(1.0, thrust_hist.max() * 1.2))
    (l_thrust,) = ax_t.plot([], [], "b-", lw=0.5, alpha=0.5, label="T")
    ax_d.legend(fontsize=5, loc="upper right")
    ax_t.legend(fontsize=5, loc="lower right")

    anim = SimAnimator("geometric_control", out_dir=Path(__file__).parent)
    anim._fig = viz.fig
    title = viz.ax3d.set_title("Geometric SO(3)")

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        ref_3d.set_data([refs[k, 0]], [refs[k, 1]])
        ref_3d.set_3d_properties([refs[k, 2]])
        ref_top.set_data([refs[k, 0]], [refs[k, 1]])
        l_err.set_data(times[:k], err[:k])
        l_thrust.set_data(times[:k], thrust_hist[:k])
        title.set_text(f"Geometric SO(3) — t={times[k]:.1f}s  err={err[k]:.2f}m")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
