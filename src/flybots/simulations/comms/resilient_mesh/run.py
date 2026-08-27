# Erwin Lejeune - 2026-08-27
"""Resilient mesh: what happens when one relay dies.

A connected network is not a survivable one. A chain has healthy algebraic
connectivity right up until any single agent in it fails, at which point it
is two networks. `degree_of_connectivity` measures that directly, by
removal rather than by a spectral proxy: k = 1 means one loss splits the
mesh, k = 2 means it survives any single loss.

Two fleets fly the same task with the same controller and different
connectivity floors, and at the halfway mark the most heavily loaded agent
in each -- the one whose removal costs the most connectivity -- is switched
off.

Neither fleet actually splits, and that is worth stating rather than
staging. The tightly-held fleet keeps a worst-case k of 2 or better, so it
still survives any single further loss; the loosely-held one sits at k = 1
before the failure and after it, having been a single point of failure the
whole time. Redundancy is what is being measured, and it is a different
question from "is the network up".

Note also that k is computed on the *thresholded* graph, while lambda-2 is
computed on the weighted one. A Gaussian link never reaches exactly zero,
so lambda-2 stays strictly positive even for a fleet that has no usable
links left -- it merely becomes very small. The two metrics answer
different questions and the threshold matters: at 0.25 a link here needs
50 m and the median pair sits 70 m apart, which reads as a disconnected
mesh that is doing fine.

References:
- M. Fiedler, "Algebraic connectivity of graphs," Czechoslovak Mathematical
  Journal, 1973.
- L. Sabattini et al., "Decentralized connectivity maintenance for
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

WORLD = 260.0
ALT = 40.0
N_AGENTS = 14
SIGMA = 30.0
LINK_THRESHOLD = 0.10
DT = 0.05
STEPS = 1600
FAILURE_STEP = STEPS // 2


def _worst_single_loss(weights: np.ndarray, alive: np.ndarray) -> int:
    """Index of the agent whose removal costs the most connectivity.

    Failing a random agent mostly picks one nobody depended on, which
    proves nothing. This is the adversarial choice.
    """
    live = np.flatnonzero(alive)
    worst, worst_lambda = int(live[0]), np.inf
    for idx in live:
        keep = live[live != idx]
        if len(keep) < 2:
            continue
        lam = algebraic_connectivity(weights[np.ix_(keep, keep)])
        if lam < worst_lambda:
            worst, worst_lambda = int(idx), lam
    return worst


def _fly(lambda_min: float):
    rng = np.random.default_rng(21)
    link = GaussianLink(SIGMA)
    ctrl = ConnectivityController(link, lambda_min=lambda_min, gain=25.0, max_force=12.0)

    pos = rng.uniform(100.0, 160.0, (N_AGENTS, 3))
    pos[:, 2] = ALT
    goal = np.column_stack(
        [
            rng.uniform(20.0, WORLD - 20.0, N_AGENTS),
            rng.uniform(20.0, WORLD - 20.0, N_AGENTS),
            np.full(N_AGENTS, ALT),
        ]
    )
    vel = np.zeros_like(pos)
    alive = np.ones(N_AGENTS, dtype=bool)

    history = np.zeros((STEPS, N_AGENTS, 3))
    alive_hist = np.zeros((STEPS, N_AGENTS), dtype=bool)
    lam = np.zeros(STEPS)
    kconn = np.zeros(STEPS)
    failed = -1

    for step in range(STEPS):
        if step == FAILURE_STEP:
            failed = _worst_single_loss(link.weights(pos), alive)
            alive[failed] = False

        live = np.flatnonzero(alive)
        p_live, v_live = pos[live], vel[live]

        nominal = np.clip(0.9 * (goal[live] - p_live) - 1.8 * v_live, -4.0, 4.0)
        command = nominal + ctrl.forces(p_live)

        vel[live] = v_live + command * DT
        pos[live] = p_live + vel[live] * DT

        history[step] = pos
        alive_hist[step] = alive
        w = link.weights(pos[live])
        lam[step] = algebraic_connectivity(w)
        kconn[step] = degree_of_connectivity(w, LINK_THRESHOLD)

    return history, alive_hist, lam, kconn, failed


def _rolling_min(series: np.ndarray, window: int) -> np.ndarray:
    """Worst value over a trailing window.

    k flips as agents drift across the link threshold, so the raw trace
    square-waves. The worst recent value is both readable and the right
    statistic for a resilience claim: what matters is the weakest moment,
    not the average one.
    """
    out = np.empty_like(series)
    for i in range(len(series)):
        out[i] = series[max(0, i - window) : i + 1].min()
    return out


def main() -> None:
    tight, alive_t, lam_t, k_t, failed_t = _fly(lambda_min=0.55)
    loose, alive_l, lam_l, k_l, failed_l = _fly(lambda_min=0.24)
    times = np.arange(STEPS) * DT

    # Report the same statistic the plot draws, so the two agree.
    window = int(6.0 / DT)
    k_t_plot = _rolling_min(k_t, window)
    k_l_plot = _rolling_min(k_l, window)

    logger = SimLogger("resilient_mesh", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "Connectivity maintenance under agent loss")
    logger.log_metadata("n_agents", N_AGENTS)
    logger.log_metadata("failure_time_s", FAILURE_STEP * DT)
    for step in range(STEPS):
        logger.log_step(
            t=times[step],
            positions=tight[step],
            algebraic_connectivity=float(lam_t[step]),
            k_connectivity=int(k_t[step]),
            algebraic_connectivity_loose=float(lam_l[step]),
            k_connectivity_loose=int(k_l[step]),
        )
    logger.log_summary("failed_agent_tight", failed_t)
    logger.log_summary("failed_agent_loose", failed_l)
    logger.log_summary("k_worst_before_failure_tight", int(k_t_plot[FAILURE_STEP - 1]))
    logger.log_summary("k_worst_before_failure_loose", int(k_l_plot[FAILURE_STEP - 1]))
    logger.log_summary("k_worst_after_failure_tight", int(k_t_plot[-1]))
    logger.log_summary("k_worst_after_failure_loose", int(k_l_plot[-1]))
    logger.log_summary("lambda2_after_failure_tight", float(lam_t[-1]))
    logger.log_summary("lambda2_after_failure_loose", float(lam_l[-1]))
    logger.log_summary("survived_tight", bool(lam_t[-1] > 1e-3))
    logger.log_summary("survived_loose", bool(lam_l[-1] > 1e-3))
    logger.save()

    skip = max(1, STEPS // 140)
    idx = list(range(0, STEPS, skip))
    colours = plt.cm.turbo(np.linspace(0.1, 0.9, N_AGENTS))
    c_rgb = [c[:3] for c in colours]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_lam = fig.add_subplot(gs[0, 1])
    ax_k = fig.add_subplot(gs[1, 1])

    fig.suptitle("Resilient Mesh — one relay is switched off at the halfway mark", fontsize=13)

    ax3d.set_xlim(0, WORLD)
    ax3d.set_ylim(0, WORLD)
    ax3d.set_zlim(0, 120)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")

    ax_lam.set_xlim(0, times[-1])
    ax_lam.set_yscale("log")
    ax_lam.set_ylim(1e-6, 3.0)
    ax_lam.set_xlabel("Time [s]", fontsize=8)
    ax_lam.set_ylabel(r"$\lambda_2$", fontsize=8)
    ax_lam.set_title("Connectivity across the failure", fontsize=9)
    ax_lam.grid(True, alpha=0.3, which="both")
    ax_lam.axvline(FAILURE_STEP * DT, color="k", ls=":", lw=1.2, label="agent lost")
    (l_t,) = ax_lam.plot([], [], color="tab:green", lw=1.5, label="tight floor")
    (l_l,) = ax_lam.plot([], [], color="tab:red", lw=1.3, label="loose floor")
    ax_lam.legend(fontsize=7, loc="lower left")

    ax_k.set_xlim(0, times[-1])
    ax_k.set_ylim(-0.2, 3.5)
    ax_k.set_xlabel("Time [s]", fontsize=8)
    ax_k.set_ylabel("k-connectivity", fontsize=8)
    ax_k.set_title("k = 1 means one loss splits the mesh (worst over 6 s)", fontsize=9)
    ax_k.grid(True, alpha=0.3)
    ax_k.axvline(FAILURE_STEP * DT, color="k", ls=":", lw=1.2)
    (k_tl,) = ax_k.plot([], [], color="tab:green", lw=1.5, label="tight floor")
    (k_ll,) = ax_k.plot([], [], color="tab:red", lw=1.3, label="loose floor")
    ax_k.legend(fontsize=7, loc="upper right")

    veh: list = []
    links: list = []
    title = ax3d.set_title("t = 0.0 s")
    link_model = GaussianLink(SIGMA)

    anim = SimAnimator("resilient_mesh", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(frame: int) -> None:
        step = idx[frame]
        p, alive = tight[step], alive_t[step]

        clear_vehicle_artists(veh)
        for artist in links:
            artist.remove()
        links.clear()

        live = np.flatnonzero(alive)
        w = link_model.weights(p[live])
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                if w[a, b] > LINK_THRESHOLD:
                    i, j = live[a], live[b]
                    (ln,) = ax3d.plot(
                        [p[i, 0], p[j, 0]],
                        [p[i, 1], p[j, 1]],
                        [p[i, 2], p[j, 2]],
                        color="tab:cyan",
                        lw=0.6,
                        alpha=0.55,
                    )
                    links.append(ln)

        for i in range(N_AGENTS):
            if not alive[i]:
                # Draw the dead agent where it stopped, in grey.
                (x,) = ax3d.plot([p[i, 0]], [p[i, 1]], [p[i, 2]], "x", color="0.5", ms=9, mew=2)
                links.append(x)
                continue
            v = (tight[step, i] - tight[step - 1, i]) / DT if step > 0 else np.zeros(3)
            veh.extend(
                draw_quadrotor_3d(
                    ax3d,
                    p[i],
                    attitude_from_velocity(v),
                    size=5.0,
                    arm_colors=(c_rgb[i], c_rgb[i]),
                    center_color=c_rgb[i],
                    motor_color=c_rgb[i],
                )
            )

        l_t.set_data(times[:step], np.maximum(lam_t[:step], 1e-6))
        l_l.set_data(times[:step], np.maximum(lam_l[:step], 1e-6))
        k_tl.set_data(times[:step], k_t_plot[:step])
        k_ll.set_data(times[:step], k_l_plot[:step])
        state = "before failure" if step < FAILURE_STEP else "after failure"
        title.set_text(f"t = {step * DT:.1f} s   {state}   k = {int(k_t[step])}")

    anim.animate(update, len(idx))
    anim.save()

    for label, k, lam in (("tight floor", k_t_plot, lam_t), ("loose floor", k_l_plot, lam_l)):
        print(
            f"  {label}: worst k {int(k[FAILURE_STEP - 1])} -> {int(k[-1])}"
            f"   lambda2 after {lam[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
