# Erwin Lejeune - 2026-02-19
"""Virtual-structure formation: 100m env with quad models, 3D + top + data.

4 agents track a moving virtual structure. The structure follows the
shared swarm figure-8, so the formation has to hold its shape through a
reversal rather than settling into one steady bank.

Reference: M. A. Lewis, K.-H. Tan, "High Precision Formation Control of Mobile
Robots Using Virtual Structures," Autonomous Robots, 1997. DOI: 10.1023/A:1008814708459
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from flybots.logging import SimLogger
from flybots.simulations.standards import swarm_figure_8_reference
from flybots.swarm.virtual_structure import VirtualStructure
from flybots.vehicles.multirotor.quadrotor import Quadrotor
from flybots.visualization import SimAnimator
from flybots.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 100.0


def main() -> None:
    n_ag = 4
    offsets = np.array([[10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0.0]])
    ctrl = VirtualStructure(body_offsets=offsets, kp=2.0, kd=1.5)
    rng = np.random.default_rng(2)
    pos = np.array([[50, 50, 50.0]] * n_ag) + rng.uniform(-5, 5, (n_ag, 3))
    vel = np.zeros((n_ag, 3))
    dt, n_steps = 0.1, 300

    snap = [pos.copy()]
    form_err = np.zeros((n_steps, n_ag))
    centroid_hist = np.zeros((n_steps, 3))
    mean_dist = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt
        # The virtual body flies the shared swarm figure-8, and the
        # reference supplies its own analytic derivatives: driven on
        # position alone the whole formation trails the body by kd*v/kp.
        body_pos, body_vel, body_acc = swarm_figure_8_reference(t)
        # The structure is a rigid body: yaw it along the path tangent so
        # the formation banks into each turn instead of translating
        # sideways through the crossing.
        body_yaw = float(np.arctan2(body_vel[1], body_vel[0]))
        # d/dt atan2(vy, vx). Constant on an orbit, but not on a figure-8:
        # it reverses sign at the crossing, and feeding the old constant
        # would yaw the structure the wrong way through half the path.
        speed_sq = float(body_vel[0] ** 2 + body_vel[1] ** 2)
        body_yaw_rate = (
            float(body_vel[0] * body_acc[1] - body_vel[1] * body_acc[0]) / speed_sq
            if speed_sq > 1e-9
            else 0.0
        )
        centroid_hist[step] = body_pos
        forces = ctrl.compute_forces(
            pos,
            vel,
            body_pos,
            body_yaw=body_yaw,
            body_vel=body_vel,
            body_yaw_rate=body_yaw_rate,
            body_acc=body_acc,
        )
        # Double integrator: the PD's own kd term is the damping. Adding a
        # multiplicative velocity decay on top acts like unmodelled drag
        # and leaves a standing formation error of kd·v_body/kp.
        vel = vel + forces * dt
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(speed > 8.0, vel / speed * 8.0, vel)
        pos = pos + vel * dt
        snap.append(pos.copy())
        des = ctrl.desired_positions(body_pos, body_yaw)
        for i in range(n_ag):
            form_err[step, i] = np.linalg.norm(pos[i] - des[i])
        dists = [np.linalg.norm(pos[i] - pos[j]) for i in range(n_ag) for j in range(i + 1, n_ag)]
        mean_dist[step] = np.mean(dists)

    times = np.arange(n_steps) * dt

    logger = SimLogger("virtual_structure", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "Virtual Structure")
    logger.log_metadata("n_agents", n_ag)
    logger.log_metadata("dt", dt)
    logger.log_metadata("n_steps", n_steps)
    for step in range(n_steps):
        logger.log_step(
            t=times[step],
            positions=snap[step],
            formation_error=form_err[step].tolist(),
            mean_neighbor_dist=mean_dist[step],
        )
    logger.log_summary("mean_formation_error_m", float(form_err.mean()))
    logger.log_summary("final_formation_error_m", float(form_err[-1].mean()))
    logger.save()

    skip = max(1, n_steps // 100)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.Set2(np.linspace(0, 1, n_ag))
    c_rgb = [c[:3] for c in colors]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1])

    fig.suptitle("Virtual Structure Formation (100m env)", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_top.set_xlim(0, WORLD_SIZE)
    ax_top.set_ylim(0, WORLD_SIZE)
    ax_top.set_aspect("equal")
    ax_top.set_title("Top Down", fontsize=9)
    ax_top.grid(True, alpha=0.15)

    ax_err.set_xlim(0, n_steps * dt)
    ax_err.set_ylim(0, max(1.0, form_err.max() * 1.1))
    ax_err.set_xlabel("Time [s]", fontsize=8)
    ax_err.set_ylabel("Formation Error [m]", fontsize=8)
    ax_err.set_title("Formation Error", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    err_lines = [
        ax_err.plot([], [], color=colors[i], lw=0.8, label=f"A{i}")[0] for i in range(n_ag)
    ]
    ax_err.legend(fontsize=6, ncol=2)

    ax_dist.set_xlim(0, n_steps * dt)
    ax_dist.set_ylim(0, max(10, mean_dist.max() * 1.2))
    ax_dist.set_xlabel("Time [s]", fontsize=8)
    ax_dist.set_ylabel("Mean Neighbor Distance [m]", fontsize=8)
    ax_dist.set_title("Neighbor Distances", fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    (ldist,) = ax_dist.plot([], [], "b-", lw=0.8)

    (centroid_top,) = ax_top.plot(
        [],
        [],
        "r*",
        ms=12,
        zorder=10,
        label="Virtual Centre",
    )
    sc_top = ax_top.scatter(pos[:, 0], pos[:, 1], c=colors, s=30, zorder=5)
    trails_top = [ax_top.plot([], [], color=colors[i], lw=0.4, alpha=0.3)[0] for i in range(n_ag)]
    ax_top.legend(fontsize=7)

    veh_arts: list = []
    title = ax3d.set_title("t = 0.0 s")
    all_snap = np.array(snap)

    anim = SimAnimator("virtual_structure", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        step = idx[f]
        p = snap[step]
        c = centroid_hist[min(step, n_steps - 1)]

        clear_vehicle_artists(veh_arts)
        for i in range(n_ag):
            R = Quadrotor.rotation_matrix(0, 0, 0)
            veh_arts.extend(
                draw_quadrotor_3d(
                    ax3d,
                    p[i],
                    R,
                    size=3.0,
                    arm_colors=(c_rgb[i], c_rgb[i]),
                    center_color=c_rgb[i],
                    motor_color=c_rgb[i],
                )
            )

        centroid_top.set_data([c[0]], [c[1]])
        sc_top.set_offsets(p[:, :2])

        for i in range(n_ag):
            trail = all_snap[: step + 1 : max(1, step // 50), i]
            trails_top[i].set_data(trail[:, 0], trail[:, 1])
            err_lines[i].set_data(times[:step], form_err[:step, i])

        ldist.set_data(times[:step], mean_dist[:step])
        title.set_text(f"Virtual Structure — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
