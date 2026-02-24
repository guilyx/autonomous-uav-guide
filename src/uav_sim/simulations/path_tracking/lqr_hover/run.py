# Erwin Lejeune - 2026-02-15
"""LQR hover stabilisation: 3-panel live simulation.

Shows a quadrotor recovering from an offset position to a hover target
using a continuous-time LQR controller, rendered across 3D, top-down,
and data views with urban buildings.

Reference: B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear Quadratic
Methods," Prentice-Hall, 1990.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.logging import SimLogger
from uav_sim.path_tracking.flight_ops import init_hover
from uav_sim.path_tracking.lqr_controller import LQRController
from uav_sim.simulations.common import STANDARD_DURATION, WORLD_SIZE, frame_indices
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")


def main() -> None:
    world, buildings = default_world()

    target_pos = np.array([15.0, 15.0, 12.0])
    start = target_pos + np.array([-1.0, -1.0, -0.5])
    target_state = np.zeros(12)
    target_state[:3] = target_pos

    quad = Quadrotor()
    quad.reset(position=start)
    init_hover(quad)

    ctrl = LQRController(
        mass=quad.params.mass,
        gravity=quad.params.gravity,
        inertia=quad.params.inertia,
    )

    dt, duration = 0.01, STANDARD_DURATION
    steps = int(duration / dt)
    states = np.zeros((steps, 12))
    controls = np.zeros((steps, 4))
    times = np.zeros(steps)

    rng = np.random.default_rng(42)
    act_noise_std = np.array([0.3, 0.01, 0.01, 0.005])
    wind_base = rng.uniform(-0.3, 0.3, size=3)
    wind_base[2] = 0.0

    for i in range(steps):
        states[i] = quad.state
        times[i] = i * dt
        u = ctrl.compute(quad.state, target_state)
        u += rng.normal(0, act_noise_std)
        controls[i] = u
        wind_gust = wind_base + rng.normal(0, 0.15, size=3)
        wind_gust[2] = 0.0
        wind_wrench = np.zeros(4)
        wind_wrench[0] = np.dot(wind_gust, np.array([0, 0, 1])) * quad.params.mass
        quad.state[6:9] += wind_gust * dt
        quad.step(u, dt)

    pos = states[:, :3]
    err = np.linalg.norm(pos - target_pos, axis=1)

    logger = SimLogger("lqr_hover", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "LQR")
    logger.log_metadata("dt", dt)
    logger.log_metadata("duration", duration)
    for i in range(steps):
        logger.log_step(
            t=times[i],
            position=pos[i],
            velocity=states[i, 6:9],
            tracking_error=err[i],
            wrench=controls[i],
        )
    logger.log_summary("mean_error_m", float(err.mean()))
    logger.log_summary("max_error_m", float(err.max()))
    logger.log_summary("final_error_m", float(err[-1]))
    logger.save()

    # ── visualisation ──────────────────────────────────────────────────
    idx = frame_indices(steps)
    n_frames = len(idx)

    viz = ThreePanelViz(title="LQR Hover Stabilisation", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.mark_start_goal(start, target_pos)

    trail = viz.create_trail_artists()

    ax_d = viz.setup_data_axes(title="Position Error [m]", ylabel="Error")
    ax_d.set_xlim(0, duration)
    ax_d.set_ylim(0, max(0.5, err.max() * 1.1))
    (l_err,) = ax_d.plot([], [], "r-", lw=0.8, label="||e||")
    ax_att = ax_d.twinx()
    ax_att.set_ylabel("Attitude [rad]", fontsize=7)
    ax_att.tick_params(labelsize=6)
    (l_phi,) = ax_att.plot([], [], "c-", lw=0.5, alpha=0.6, label="φ")
    (l_theta,) = ax_att.plot([], [], "m-", lw=0.5, alpha=0.6, label="θ")
    ax_att.set_ylim(-0.3, 0.3)
    ax_d.legend(fontsize=5, loc="upper right")
    ax_att.legend(fontsize=5, loc="lower right")

    anim = SimAnimator("lqr_hover", out_dir=Path(__file__).parent)
    anim._fig = viz.fig
    title = viz.ax3d.set_title("LQR Hover")

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        l_err.set_data(times[:k], err[:k])
        l_phi.set_data(times[:k], states[:k, 3])
        l_theta.set_data(times[:k], states[:k, 4])
        title.set_text(f"LQR Hover — t={times[k]:.1f}s  err={err[k]:.3f}m")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
