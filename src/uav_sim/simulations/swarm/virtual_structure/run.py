# Erwin Lejeune - 2026-02-17
"""Virtual-structure formation: multi-panel with 3D + formation error.

Shows 4 agents tracking a moving virtual structure with coloured trails
and per-agent position error converging to zero.

Reference: M. A. Lewis, K.-H. Tan, "High Precision Formation Control of Mobile
Robots Using Virtual Structures," Autonomous Robots, 1997. DOI: 10.1023/A:1008814708459
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.virtual_structure import VirtualStructure
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def main() -> None:
    n_ag = 4
    offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0.0]])
    ctrl = VirtualStructure(body_offsets=offsets, kp=3.0, kd=2.0)
    rng = np.random.default_rng(2)
    pos = np.zeros((n_ag, 3)) + rng.uniform(-1, 1, (n_ag, 3))
    vel = np.zeros((n_ag, 3))
    dt, n_steps = 0.05, 300

    snap = [pos.copy()]
    form_err = np.zeros((n_steps, n_ag))
    centroid_hist = np.zeros((n_steps, 3))
    for step in range(n_steps):
        t = step * dt
        body_pos = np.array([2 * np.sin(0.3 * t), 2 * np.cos(0.3 * t), 1.0])
        centroid_hist[step] = body_pos
        forces = ctrl.compute_forces(pos, vel, body_pos, body_yaw=0.0)
        vel = vel + forces * dt
        pos = pos + vel * dt
        snap.append(pos.copy())
        for i in range(n_ag):
            form_err[step, i] = np.linalg.norm(pos[i] - body_pos - offsets[i])

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.Set2(np.linspace(0, 1, n_ag))

    anim = SimAnimator("virtual_structure", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(13, 6))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax_err = fig.add_subplot(gs[1])
    fig.suptitle("Virtual Structure Formation", fontsize=13)

    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax3d, all_p, pad=2)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=50, depthshade=True)
    (centroid_dot,) = ax3d.plot([], [], [], "r*", ms=12, label="Virtual Centre")
    trails_3d = []
    all_snap = np.array(snap)
    for i in range(n_ag):
        (tr,) = ax3d.plot([], [], [], "-", color=colors[i], lw=0.8, alpha=0.5)
        trails_3d.append(tr)
    ax3d.legend(fontsize=7, loc="upper left")

    ax_err.set_xlim(0, n_steps * dt)
    ax_err.set_ylim(0, max(0.5, form_err.max() * 1.1))
    ax_err.set_xlabel("Time [s]", fontsize=9)
    ax_err.set_ylabel("Formation Error [m]", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    err_lines = []
    for i in range(n_ag):
        (ln,) = ax_err.plot([], [], color=colors[i], lw=1, label=f"Agent {i}")
        err_lines.append(ln)
    ax_err.legend(fontsize=6, ncol=2, loc="upper right")
    ax_err.tick_params(labelsize=7)

    title = ax3d.set_title("t = 0.0 s")

    def update(f):
        step = idx[f]
        p = snap[step]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        c = centroid_hist[min(step, n_steps - 1)]
        centroid_dot.set_data([c[0]], [c[1]])
        centroid_dot.set_3d_properties([c[2]])
        for i in range(n_ag):
            trail_data = all_snap[: step + 1, i]
            trails_3d[i].set_data(trail_data[:, 0], trail_data[:, 1])
            trails_3d[i].set_3d_properties(trail_data[:, 2])
            err_lines[i].set_data(times[:step], form_err[:step, i])
        title.set_text(f"Virtual Structure — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
