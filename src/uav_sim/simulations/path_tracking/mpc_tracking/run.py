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
from uav_sim.path_tracking.mpc_controller import MPCController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CENTER = np.array([15.0, 15.0])
RX, RY = 6.0, 4.0
ALT = 12.0
OMEGA = 0.12


def _ref(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Figure-8 reference: position and velocity."""
    pos = np.array(
        [
            CENTER[0] + RX * np.sin(OMEGA * t),
            CENTER[1] + RY * np.sin(2 * OMEGA * t),
            ALT + 1.5 * np.sin(0.3 * t),
        ]
    )
    vel = np.array(
        [
            RX * OMEGA * np.cos(OMEGA * t),
            RY * 2 * OMEGA * np.cos(2 * OMEGA * t),
            1.5 * 0.3 * np.cos(0.3 * t),
        ]
    )
    return pos, vel


def main() -> None:
    world, buildings = default_world()

    quad = Quadrotor()
    rp0, _ = _ref(0.0)
    quad.reset(position=rp0.copy())
    from uav_sim.path_tracking.flight_ops import init_hover

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
    wrench = quad.hover_wrench()

    ctrl_counter = 0
    for i in range(steps):
        t = i * dt
        rp, rv = _ref(t)
        refs[i] = rp
        states[i] = quad.state

        ctrl_counter += dt
        if ctrl_counter >= ctrl_dt - 1e-8:
            wrench = ctrl.compute(quad.state, rp, target_vel=rv)
            ctrl_counter = 0.0
        quad.step(wrench, dt)

    pos = states[:, :3]

    # ── Visualisation ─────────────────────────────────────────────────
    skip = max(1, steps // 120)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="MPC Path Tracking — Figure-8", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(refs, color="red", lw=1.0, alpha=0.3, label="Reference")

    trail = viz.create_trail_artists()
    (ref_dot_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10)
    (ref_dot_top,) = viz.ax_top.plot([], [], "r*", ms=8)
    (ref_dot_side,) = viz.ax_side.plot([], [], "r*", ms=8)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    anim = SimAnimator("mpc_tracking", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        ref_dot_3d.set_data([refs[k, 0]], [refs[k, 1]])
        ref_dot_3d.set_3d_properties([refs[k, 2]])
        ref_dot_top.set_data([refs[k, 0]], [refs[k, 1]])
        ref_dot_side.set_data([refs[k, 0]], [refs[k, 2]])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
