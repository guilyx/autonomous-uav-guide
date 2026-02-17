# Erwin Lejeune - 2026-02-15
"""Leader-follower: one leader on a circle, 3 followers maintain offsets.

Reference: J. Desai et al., "Modeling and Control of Formations of
Nonholonomic Mobile Robots," IEEE T-RA, 2001. DOI: 10.1109/70.976023
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.swarm.leader_follower import LeaderFollower
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0.0]])
    ctrl = LeaderFollower(offsets=offsets, kp=4.0, kd=2.0)
    N_ag = 1 + ctrl.num_followers
    pos = np.zeros((N_ag, 3))
    pos[0] = [0, 0, 1]
    vel = np.zeros((N_ag, 3))
    dt, n_steps = 0.05, 300
    snap = [pos.copy()]
    for step in range(n_steps):
        t = step * dt
        new_leader = np.array([2 * np.cos(0.4 * t), 2 * np.sin(0.4 * t), 1.0])
        leader_vel = (new_leader - pos[0]) / dt
        pos[0] = new_leader
        forces = ctrl.compute_forces(pos[0], leader_vel, pos[1:], vel[1:])
        vel[1:] = vel[1:] + forces * dt
        pos[1:] = pos[1:] + vel[1:] * dt
        snap.append(pos.copy())
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    anim = SimAnimator("leader_follower", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Leader-Follower Formation")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="steelblue", s=40)
    sc_leader = ax.scatter(
        [pos[0, 0]], [pos[0, 1]], [pos[0, 2]], c="red", s=80, marker="D", label="Leader", zorder=5
    )
    ax.legend(fontsize=8)
    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax, all_p, pad=2)

    def update(f):
        p = snap[idx[f]]
        sc._offsets3d = (p[1:, 0], p[1:, 1], p[1:, 2])
        sc_leader._offsets3d = ([p[0, 0]], [p[0, 1]], [p[0, 2]])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
