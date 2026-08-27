# Erwin Lejeune - 2026-08-27
"""Connectivity maintenance: 24 agents keep a mesh alive under a task that tears it.

Every agent is given a goal drawn at random across a 400 m box, which is far
enough apart that flying the task straight fragments the radio network into
islands. A connectivity controller ascends the gradient of algebraic
connectivity, so the fleet trades goal progress for a mesh that survives.

Both runs use the same goals and the same nominal controller, so the only
difference is whether the network is treated as part of the plant.

The metric is λ₂, the second smallest eigenvalue of the weighted graph
Laplacian, which is strictly positive exactly while the network is
connected. "Is it connected" is a boolean that tells a controller nothing
until it is already too late; λ₂ degrades smoothly and can be pushed on.

Reference: L. Sabattini et al., "Decentralized connectivity maintenance for
cooperative control of mobile robotic systems," IJRR, 2013.
DOI: 10.1177/0278364913499085
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from flybots.comms import (
    ConnectivityController,
    GaussianLink,
    algebraic_connectivity,
    degree_of_connectivity,
)
from flybots.logging import SimLogger
from flybots.visualization import SimAnimator, attitude_from_velocity
from flybots.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")

WORLD_SIZE = 400.0
CRUISE_ALT = 40.0
N_AGENTS = 24
SIGMA = 30.0
LAMBDA_MIN = 0.25
DT = 0.05
STEPS = 1400
LINK_DRAW_THRESHOLD = 0.25


def _view_window(positions: np.ndarray, half: float) -> tuple:
    """Axis limits that follow the fleet.

    The goals are scattered across the whole world on purpose -- that is
    what tears the mesh -- but the fleet the controller holds together is a
    fraction of it, so a box drawn around the world renders the network as a
    speck. Tracking the centroid keeps the mesh legible and costs nothing:
    the physics is unchanged, only the camera moves.
    """
    c = positions.mean(axis=0)
    return (
        (c[0] - half, c[0] + half),
        (c[1] - half, c[1] + half),
        (max(0.0, c[2] - half * 0.6), c[2] + half * 0.6),
    )


def _scenario():
    rng = np.random.default_rng(11)
    start = rng.uniform(140.0, 160.0, (N_AGENTS, 3))
    start[:, 2] = CRUISE_ALT
    goal = np.column_stack(
        [
            rng.uniform(0.0, WORLD_SIZE, N_AGENTS),
            rng.uniform(0.0, WORLD_SIZE, N_AGENTS),
            np.full(N_AGENTS, CRUISE_ALT),
        ]
    )
    return start, goal


def _fly(keep_connected: bool):
    start, goal = _scenario()
    link = GaussianLink(SIGMA)
    ctrl = ConnectivityController(link, lambda_min=LAMBDA_MIN, gain=25.0, max_force=12.0)

    pos = start.copy()
    vel = np.zeros_like(pos)
    history = np.zeros((STEPS, N_AGENTS, 3))
    lam = np.zeros(STEPS)
    goal_err = np.zeros(STEPS)

    for step in range(STEPS):
        nominal = np.clip(0.9 * (goal - pos) - 1.8 * vel, -4.0, 4.0)
        command = nominal + ctrl.forces(pos) if keep_connected else nominal
        vel = vel + command * DT
        pos = pos + vel * DT

        history[step] = pos
        lam[step] = algebraic_connectivity(link.weights(pos))
        goal_err[step] = float(np.mean(np.linalg.norm(pos - goal, axis=1)))

    return history, lam, goal_err, link


def main() -> None:
    aware, lam_on, err_on, link = _fly(keep_connected=True)
    naive, lam_off, err_off, _ = _fly(keep_connected=False)
    times = np.arange(STEPS) * DT

    k_aware = degree_of_connectivity(link.weights(aware[-1]), LINK_DRAW_THRESHOLD)
    k_naive = degree_of_connectivity(link.weights(naive[-1]), LINK_DRAW_THRESHOLD)

    logger = SimLogger("connectivity_maintenance", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "Algebraic connectivity maintenance")
    logger.log_metadata("n_agents", N_AGENTS)
    logger.log_metadata("sigma_m", SIGMA)
    logger.log_metadata("lambda_min", LAMBDA_MIN)
    for step in range(STEPS):
        logger.log_step(
            t=times[step],
            positions=aware[step],
            algebraic_connectivity=float(lam_on[step]),
            algebraic_connectivity_naive=float(lam_off[step]),
            goal_error=float(err_on[step]),
        )
    logger.log_summary("final_lambda2", float(lam_on[-1]))
    logger.log_summary("final_lambda2_naive", float(lam_off[-1]))
    logger.log_summary("min_lambda2", float(lam_on.min()))
    logger.log_summary("k_connectivity", k_aware)
    logger.log_summary("k_connectivity_naive", k_naive)
    logger.save()

    skip = max(1, STEPS // 140)
    idx = list(range(0, STEPS, skip))
    colours = plt.cm.viridis(np.linspace(0, 1, N_AGENTS))
    c_rgb = [c[:3] for c in colours]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_lam = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])

    fig.suptitle("Connectivity Maintenance — 24 agents, goals scattered over 400 m", fontsize=13)

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")

    ax_lam.set_xlim(0, times[-1])
    ax_lam.set_yscale("log")
    ax_lam.set_ylim(1e-7, 2.0)
    ax_lam.set_xlabel("Time [s]", fontsize=8)
    ax_lam.set_ylabel(r"$\lambda_2$", fontsize=8)
    ax_lam.set_title("Algebraic connectivity — log scale", fontsize=9)
    ax_lam.grid(True, alpha=0.3, which="both")
    ax_lam.axhline(LAMBDA_MIN, color="crimson", ls="--", lw=1.1, label="floor")
    (l_on,) = ax_lam.plot([], [], color="tab:green", lw=1.5, label="connectivity-aware")
    (l_off,) = ax_lam.plot([], [], color="tab:red", lw=1.2, label="task only")
    ax_lam.legend(fontsize=7, loc="lower left")

    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(0, max(err_on.max(), err_off.max()) * 1.1)
    ax_err.set_xlabel("Time [s]", fontsize=8)
    ax_err.set_ylabel("Mean distance to goal [m]", fontsize=8)
    ax_err.set_title("The price: goals are not reached", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    (e_on,) = ax_err.plot([], [], color="tab:green", lw=1.5, label="connectivity-aware")
    (e_off,) = ax_err.plot([], [], color="tab:red", lw=1.2, label="task only")
    ax_err.legend(fontsize=7, loc="upper right")

    veh: list = []
    links: list = []
    title = ax3d.set_title("t = 0.0 s")

    anim = SimAnimator("connectivity_maintenance", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(frame: int) -> None:
        step = idx[frame]
        p = aware[step]

        xlim, ylim, zlim = _view_window(p, 62.0)
        ax3d.set_xlim(*xlim)
        ax3d.set_ylim(*ylim)
        ax3d.set_zlim(*zlim)

        clear_vehicle_artists(veh)
        for artist in links:
            artist.remove()
        links.clear()

        # Draw the radio graph itself: the mesh is the subject, so it should
        # be visible rather than inferred from how the dots are arranged.
        w = link.weights(p)
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                if w[i, j] > LINK_DRAW_THRESHOLD:
                    (ln,) = ax3d.plot(
                        [p[i, 0], p[j, 0]],
                        [p[i, 1], p[j, 1]],
                        [p[i, 2], p[j, 2]],
                        color="tab:cyan",
                        lw=0.6,
                        alpha=float(np.clip(w[i, j], 0.15, 0.9)),
                    )
                    links.append(ln)

        for i in range(N_AGENTS):
            v = (aware[step, i] - aware[step - 1, i]) / DT if step > 0 else np.zeros(3)
            veh.extend(
                draw_quadrotor_3d(
                    ax3d,
                    p[i],
                    attitude_from_velocity(v),
                    size=3.2,
                    arm_colors=(c_rgb[i], c_rgb[i]),
                    center_color=c_rgb[i],
                    motor_color=c_rgb[i],
                )
            )

        l_on.set_data(times[:step], np.maximum(lam_on[:step], 1e-7))
        l_off.set_data(times[:step], np.maximum(lam_off[:step], 1e-7))
        e_on.set_data(times[:step], err_on[:step])
        e_off.set_data(times[:step], err_off[:step])
        title.set_text(
            f"t = {step * DT:.1f} s    $\\lambda_2$ = {lam_on[step]:.3f} "
            f"(task only: {lam_off[step]:.2e})"
        )

    anim.animate(update, len(idx))
    anim.save()

    print(f"  connectivity-aware  final lambda2 {lam_on[-1]:.4f}   k = {k_aware}")
    print(f"  task only           final lambda2 {lam_off[-1]:.3e}   k = {k_naive}")


if __name__ == "__main__":
    main()
