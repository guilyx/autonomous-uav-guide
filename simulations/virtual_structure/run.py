# Erwin Lejeune - 2026-02-15
"""Virtual-structure formation following a moving centroid.

Reference: M. A. Lewis, K.-H. Tan, "High Precision Formation Control of Mobile
Robots Using Virtual Structures," Autonomous Robots, 1997. DOI: 10.1023/A:1008814708459
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.swarm.virtual_structure import VirtualStructure
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    N_ag = 4
    offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0.0]])
    ctrl = VirtualStructure(body_offsets=offsets, kp=3.0, kd=2.0)
    rng = np.random.default_rng(2)
    pos = np.zeros((N_ag, 3)) + rng.uniform(-1, 1, (N_ag, 3))
    vel = np.zeros((N_ag, 3))
    dt, n_steps = 0.05, 300
    snap = [pos.copy()]
    for step in range(n_steps):
        t = step * dt
        body_pos = np.array([2 * np.sin(0.3 * t), 2 * np.cos(0.3 * t), 1.0])
        forces = ctrl.compute_forces(pos, vel, body_pos, body_yaw=0.0)
        vel = vel + forces * dt
        pos = pos + vel * dt
        snap.append(pos.copy())
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    anim = SimAnimator("virtual_structure", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Virtual Structure Formation")
    sc = ax.scatter([], [], [], c="purple", s=40)
    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax, all_p, pad=2)

    def update(f):
        p = snap[idx[f]]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
