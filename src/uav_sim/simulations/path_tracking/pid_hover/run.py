# Erwin Lejeune - 2026-02-15
"""PID-controlled hover: 3-panel live simulation.

Shows a quadrotor taking off and hovering at a target point inside an
urban environment using the cascaded PID control stack and state machine.

Reference: L. R. G. Carrillo et al., "Quad Rotorcraft Control," Springer, 2013.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.control import StateManager
from uav_sim.environment import default_world
from uav_sim.simulations.common import WORLD_SIZE, frame_indices
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

TARGET = np.array([15.0, 15.0, 12.0])


def main() -> None:
    _, buildings = default_world()

    quad = Quadrotor()
    quad.reset(position=np.array([2.0, 2.0, 0.0]))

    sm = StateManager(quad)
    sm.arm()
    sm.run_takeoff(altitude=TARGET[2], dt=0.005, timeout=10.0)
    sm.fly_to(TARGET, dt=0.005, threshold=0.5, timeout=15.0)

    for _ in range(int(5.0 / 0.005)):
        sm.step(0.005)

    states = np.array(sm.states)
    pos = states[:, :3]
    dt = 0.005
    times = np.arange(len(states)) * dt
    err = np.linalg.norm(pos - TARGET, axis=1)

    idx = frame_indices(len(states))
    n_frames = len(idx)

    viz = ThreePanelViz(title="PID Hover — Live Simulation", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.ax3d.scatter(*TARGET, c="lime", s=100, marker="*", zorder=5, label="Target")
    viz.ax_top.plot(TARGET[0], TARGET[1], "g*", ms=12, zorder=5)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    trail = viz.create_trail_artists()

    ax_d = viz.setup_data_axes(title="Position Error [m]", ylabel="Error")
    ax_d.set_xlim(0, times[-1])
    ax_d.set_ylim(0, max(1.0, err.max() * 1.1))
    (l_err,) = ax_d.plot([], [], "r-", lw=0.8, label="||e||")
    ax_z = ax_d.twinx()
    ax_z.set_ylabel("Altitude [m]", fontsize=7)
    ax_z.tick_params(labelsize=6)
    ax_z.set_ylim(0, max(15.0, pos[:, 2].max() * 1.2))
    (l_z,) = ax_z.plot([], [], "b-", lw=0.5, alpha=0.6, label="z")
    ax_d.legend(fontsize=5, loc="upper right")
    ax_z.legend(fontsize=5, loc="lower right")

    anim = SimAnimator("pid_hover", out_dir=Path(__file__).parent)
    anim._fig = viz.fig
    title = viz.ax3d.set_title("PID Hover")

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        l_err.set_data(times[:k], err[:k])
        l_z.set_data(times[:k], pos[:k, 2])
        title.set_text(f"PID Hover — t={times[k]:.1f}s  err={err[k]:.2f}m")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
