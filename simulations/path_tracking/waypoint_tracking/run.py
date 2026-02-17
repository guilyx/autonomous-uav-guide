# Erwin Lejeune - 2026-02-15
"""3-D waypoint tracking with geometric controller.

Reference: T. Lee et al., "Geometric Tracking Control of a Quadrotor UAV on
SE(3)," CDC, 2010. DOI: 10.1109/CDC.2010.5717652
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.path_tracking.geometric_controller import GeometricController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator


def main() -> None:
    quad = Quadrotor()
    quad.reset(position=np.zeros(3))
    ctrl = GeometricController()
    wps = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1.5], [0, 1, 1], [0, 0, 1.0]])
    dt, hold = 0.002, 2.0
    hold_steps = int(hold / dt)
    all_s: list[np.ndarray] = []
    for wp in wps:
        for _ in range(hold_steps):
            quad.step(ctrl.compute(quad.state, wp), dt)
            all_s.append(quad.state.copy())
    pos = np.array([s[:3] for s in all_s])
    skip = max(1, len(pos) // 250)
    idx = list(range(0, len(pos), skip))
    anim = SimAnimator("waypoint_tracking", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Waypoint Tracking (Geometric)")
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c="r", s=60, marker="D", label="WP", zorder=5)
    (trail,) = ax.plot([], [], [], "b-", lw=1.2, alpha=0.7)
    (dot,) = ax.plot([], [], [], "ko", ms=5)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_zlim(-0.1, 1.8)
    ax.legend(loc="upper left", fontsize=8)

    def update(f):
        k = idx[f]
        trail.set_data(pos[:k, 0], pos[:k, 1])
        trail.set_3d_properties(pos[:k, 2])
        dot.set_data([pos[k, 0]], [pos[k, 1]])
        dot.set_3d_properties([pos[k, 2]])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
