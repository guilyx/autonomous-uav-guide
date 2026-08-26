# Erwin Lejeune - 2026-02-19
"""Leader-follower: 100m env with quad models, 3D + top + data.

Shows the leader on the shared swarm figure-8 with 3 followers holding
offsets through the crossing.
Uses quad models, 3D/top panels plus follower error + distances.

Reference: J. Desai et al., "Modeling and Control of Formations of
Nonholonomic Mobile Robots," IEEE T-RA, 2001. DOI: 10.1109/70.976023
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.logging import SimLogger
from uav_sim.simulations.common import swarm_figure_8_ref
from uav_sim.swarm.leader_follower import LeaderFollower
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 100.0
CRUISE_ALT = 50.0


def main() -> None:
    offsets = np.array([[8, 0, 0], [-8, 0, 0], [0, 8, 0.0]])
    ctrl = LeaderFollower(offsets=offsets, kp=3.0, kd=2.0)
    n_ag = 1 + ctrl.num_followers
    # Seeded: the global np.random state is shared with everything else in
    # the process, so drawing from it makes the run irreproducible.
    rng = np.random.default_rng(11)
    pos = np.zeros((n_ag, 3))
    # Start the leader on the curve so step 0 is not a jump.
    pos[0] = swarm_figure_8_ref(0.0)[0]
    for i in range(ctrl.num_followers):
        pos[1 + i] = pos[0] + offsets[i] + rng.normal(0, 1, 3) * 15
    vel = np.zeros((n_ag, 3))
    dt, n_steps = 0.1, 300

    snap = [pos.copy()]
    follower_err = np.zeros((n_steps, ctrl.num_followers))
    mean_dist = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt
        # The leader flies the shared swarm figure-8. An orbit lets the
        # followers settle into one steady bank and hold it forever; the
        # crossing forces the formation through a reversal, which is where
        # a trailing-offset scheme actually has to work for its living.
        #
        # The reference returns its own analytic derivative. Differencing
        # against the initial position produced a 250 m/s spike on step 0
        # and kicked every follower out of formation.
        new_leader, leader_vel = swarm_figure_8_ref(t)
        pos[0] = new_leader
        forces = ctrl.compute_forces(pos[0], leader_vel, pos[1:], vel[1:])
        # Double integrator: kd in the PD is the damping. A multiplicative
        # decay on top behaves like drag and leaves the followers trailing
        # their slots by a constant offset.
        vel[1:] = vel[1:] + forces * dt
        pos[1:] = pos[1:] + vel[1:] * dt
        snap.append(pos.copy())
        for fi in range(ctrl.num_followers):
            desired = pos[0] + offsets[fi]
            follower_err[step, fi] = np.linalg.norm(pos[1 + fi] - desired)
        dists = [np.linalg.norm(pos[i] - pos[j]) for i in range(n_ag) for j in range(i + 1, n_ag)]
        mean_dist[step] = np.mean(dists)

    times = np.arange(n_steps) * dt

    logger = SimLogger("leader_follower", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "Leader-Follower")
    logger.log_metadata("n_agents", n_ag)
    logger.log_metadata("dt", dt)
    logger.log_metadata("n_steps", n_steps)
    for step in range(n_steps):
        logger.log_step(
            t=times[step],
            positions=snap[step],
            mean_follower_error=float(follower_err[step].mean()),
            mean_neighbor_dist=mean_dist[step],
        )
    logger.log_summary("mean_follower_error_m", float(follower_err.mean()))
    logger.log_summary("final_follower_error_m", float(follower_err[-1].mean()))
    logger.save()

    skip = max(1, n_steps // 100)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    agent_colors = ["red", "steelblue", "green", "orange"]
    cmap_colors = [np.array(matplotlib.colors.to_rgba(c)[:3]) for c in agent_colors]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1])

    fig.suptitle("Leader-Follower Formation (100m env)", fontsize=13)

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
    ax_err.set_ylim(0, max(1.0, follower_err.max() * 1.1))
    ax_err.set_xlabel("Time [s]", fontsize=8)
    ax_err.set_ylabel("Follower Error [m]", fontsize=8)
    ax_err.set_title("Formation Error", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    ferr_lines = []
    for fi in range(ctrl.num_followers):
        (ln,) = ax_err.plot([], [], color=agent_colors[1 + fi], lw=1, label=f"F{fi}")
        ferr_lines.append(ln)
    ax_err.legend(fontsize=7)

    ax_dist.set_xlim(0, n_steps * dt)
    ax_dist.set_ylim(0, max(5, mean_dist.max() * 1.2))
    ax_dist.set_xlabel("Time [s]", fontsize=8)
    ax_dist.set_ylabel("Mean Neighbor Distance [m]", fontsize=8)
    ax_dist.set_title("Neighbor Distances", fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    (ldist,) = ax_dist.plot([], [], "b-", lw=0.8)

    trails_top = [
        ax_top.plot([], [], color=agent_colors[i], lw=0.6, alpha=0.4)[0] for i in range(n_ag)
    ]

    veh_arts: list = []
    title = ax3d.set_title("t = 0.0 s")
    all_snap = np.array(snap)

    anim = SimAnimator("leader_follower", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        step = idx[f]
        p = snap[step]

        clear_vehicle_artists(veh_arts)
        for ai in range(n_ag):
            R = Quadrotor.rotation_matrix(0, 0, 0)
            veh_arts.extend(
                draw_quadrotor_3d(
                    ax3d,
                    p[ai],
                    R,
                    size=3.0,
                    arm_colors=(cmap_colors[ai], cmap_colors[ai]),
                    center_color=cmap_colors[ai],
                    motor_color=cmap_colors[ai],
                )
            )

        for ai in range(n_ag):
            trail = all_snap[: step + 1 : max(1, step // 50), ai]
            trails_top[ai].set_data(trail[:, 0], trail[:, 1])

        for fi in range(ctrl.num_followers):
            ferr_lines[fi].set_data(times[:step], follower_err[:step, fi])
        ldist.set_data(times[:step], mean_dist[:step])

        title.set_text(f"Leader-Follower — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
