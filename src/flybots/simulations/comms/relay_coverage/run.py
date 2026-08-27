# Erwin Lejeune - 2026-08-27
"""Relay coverage: 18 agents cover ground while staying reachable from a base.

Agent 0 is a fixed base station. The rest spread out to watch as much ground
as possible, but ground watched by an aircraft that cannot report it is not
covered, so every agent must hold a multi-hop radio path home. Coverage
wants the fleet spread; connectivity wants it clustered. The two pull
against each other directly, and the trade-off is provably NP-hard in
general, so this is a gradient heuristic rather than an optimum.

The comparison run drops the tether and lets the fleet spread freely. It
achieves roughly twice the *naive* coverage -- and almost none of it counts,
because the fleet has flown apart and only the base can still report. That
gap between naive and connected coverage is the whole point.

References:
- J. Scherer & B. Rinner, "Long-term area coverage and radio relay
  positioning using swarms of UAVs," arXiv:1810.12383, 2018.
- Y. Wang et al., "CARA: Connectivity-Aware Relay Algorithm for Multi-Robot
  Expeditions," Sensors, 2022. DOI: 10.3390/s22239140
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from flybots.comms import GaussianLink, RelayCoverageController, hop_counts
from flybots.logging import SimLogger
from flybots.visualization import SimAnimator
from flybots.visualization.vehicle_artists import draw_quadrotor_2d

matplotlib.use("Agg")

WORLD_SIZE = 300.0
CRUISE_ALT = 40.0
N_AGENTS = 18
SIGMA = 34.0
SENSING_RADIUS = 25.0
LINK_THRESHOLD = 0.5
DT = 0.05
STEPS = 2400
BASE = np.array([150.0, 150.0, CRUISE_ALT])


def _fly(tether_gain: float):
    link = GaussianLink(SIGMA)
    ctrl = RelayCoverageController(
        link,
        spread_gain=60.0 if tether_gain > 0 else 200.0,
        anchor_gain=0.5 if tether_gain > 0 else 1.5,
        tether_gain=tether_gain,
        link_threshold=LINK_THRESHOLD,
        max_force=5.0,
    )
    rng = np.random.default_rng(3)
    pos = np.zeros((N_AGENTS, 3))
    pos[0] = BASE
    pos[1:] = BASE + rng.uniform(-12.0, 12.0, (N_AGENTS - 1, 3))
    pos[:, 2] = CRUISE_ALT
    vel = np.zeros_like(pos)

    history = np.zeros((STEPS, N_AGENTS, 3))
    coverage = np.zeros(STEPS)
    naive_coverage = np.zeros(STEPS)
    reachable = np.zeros(STEPS)

    axis = np.arange(0.0, WORLD_SIZE, 6.0)
    gx, gy = np.meshgrid(axis, axis)
    cells = np.column_stack([gx.ravel(), gy.ravel()])

    for step in range(STEPS):
        command = ctrl.forces(pos)
        vel = (vel + command * DT) * 0.92
        pos = pos + vel * DT
        pos[0] = BASE
        pos[:, 2] = CRUISE_ALT

        history[step] = pos
        hops = hop_counts(link.weights(pos), 0, LINK_THRESHOLD)
        reachable[step] = float(np.isfinite(hops).sum())

        dist = np.linalg.norm(cells[:, None, :] - pos[None, :, :2], axis=2)
        naive_coverage[step] = float(np.mean(np.min(dist, axis=1) <= SENSING_RADIUS))
        live = np.isfinite(hops)
        coverage[step] = (
            float(np.mean(np.min(dist[:, live], axis=1) <= SENSING_RADIUS)) if live.any() else 0.0
        )

    return history, coverage, naive_coverage, reachable, link


def main() -> None:
    relay, cov_on, naive_on, reach_on, link = _fly(tether_gain=25.0)
    free, cov_off, naive_off, reach_off, _ = _fly(tether_gain=0.0)
    times = np.arange(STEPS) * DT

    logger = SimLogger("relay_coverage", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "Relay coverage with connectivity tether")
    logger.log_metadata("n_agents", N_AGENTS)
    logger.log_metadata("sensing_radius_m", SENSING_RADIUS)
    logger.log_metadata("sigma_m", SIGMA)
    for step in range(STEPS):
        logger.log_step(
            t=times[step],
            positions=relay[step],
            connected_coverage=float(cov_on[step]),
            naive_coverage=float(naive_on[step]),
            reachable=float(reach_on[step]),
        )
    logger.log_summary("connected_coverage", float(cov_on[-1]))
    logger.log_summary("connected_coverage_untethered", float(cov_off[-1]))
    logger.log_summary("naive_coverage_untethered", float(naive_off[-1]))
    logger.log_summary("reachable", int(reach_on[-1]))
    logger.log_summary("reachable_untethered", int(reach_off[-1]))
    logger.save()

    skip = max(1, STEPS // 140)
    idx = list(range(0, STEPS, skip))
    colours = plt.cm.plasma(np.linspace(0, 0.9, N_AGENTS))
    c_rgb = [c[:3] for c in colours]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ax_top = fig.add_subplot(gs[:, 0])
    ax_cov = fig.add_subplot(gs[0, 1])
    ax_reach = fig.add_subplot(gs[1, 1])

    fig.suptitle("Relay Coverage — spread as far as the radio allows", fontsize=13)

    ax_top.set_xlim(0, WORLD_SIZE)
    ax_top.set_ylim(0, WORLD_SIZE)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X [m]")
    ax_top.set_ylabel("Y [m]")
    ax_top.set_title("Relay chain (top-down) — circles are sensing footprints", fontsize=9)
    ax_top.grid(True, alpha=0.15)

    ax_cov.set_xlim(0, times[-1])
    ax_cov.set_ylim(0, max(naive_off.max(), cov_on.max()) * 115)
    ax_cov.set_xlabel("Time [s]", fontsize=8)
    ax_cov.set_ylabel("Coverage [%]", fontsize=8)
    ax_cov.set_title("Coverage that can actually be reported", fontsize=9)
    ax_cov.grid(True, alpha=0.3)
    (c_on,) = ax_cov.plot([], [], color="tab:green", lw=1.6, label="relay: connected")
    (c_naive,) = ax_cov.plot(
        [], [], color="tab:orange", lw=1.2, ls="--", label="untethered: naive"
    )
    (c_off,) = ax_cov.plot([], [], color="tab:red", lw=1.2, label="untethered: connected")
    ax_cov.legend(fontsize=7, loc="upper left")

    ax_reach.set_xlim(0, times[-1])
    ax_reach.set_ylim(0, N_AGENTS + 1)
    ax_reach.set_xlabel("Time [s]", fontsize=8)
    ax_reach.set_ylabel("Agents reachable from base", fontsize=8)
    ax_reach.set_title("Who can still phone home", fontsize=9)
    ax_reach.grid(True, alpha=0.3)
    (r_on,) = ax_reach.plot([], [], color="tab:green", lw=1.6, label="relay")
    (r_off,) = ax_reach.plot([], [], color="tab:red", lw=1.2, label="untethered")
    ax_reach.legend(fontsize=7, loc="lower left")

    artists: list = []
    title = ax_top.text(0.02, 0.97, "", transform=ax_top.transAxes, fontsize=9, va="top")

    anim = SimAnimator("relay_coverage", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(frame: int) -> None:
        step = idx[frame]
        p = relay[step]

        for artist in artists:
            artist.remove()
        artists.clear()

        w = link.weights(p)
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                if w[i, j] > LINK_THRESHOLD:
                    (ln,) = ax_top.plot(
                        [p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color="tab:cyan", lw=0.7, alpha=0.6
                    )
                    artists.append(ln)

        for i in range(N_AGENTS):
            circle = plt.Circle(
                (p[i, 0], p[i, 1]), SENSING_RADIUS, color=c_rgb[i], alpha=0.10, lw=0
            )
            ax_top.add_patch(circle)
            artists.append(circle)

        (base,) = ax_top.plot(p[0, 0], p[0, 1], "s", color="white", ms=11, mec="black", mew=1.4)
        artists.append(base)
        # Draw the aircraft, headed the way they are actually flying, rather
        # than marking them with dots.
        # Heading over several steps, not one: a single step at low
        # speed is mostly integrator wobble and the model spins.
        prev = relay[max(step - 12, 0)]
        for i in range(1, N_AGENTS):
            d = p[i, :2] - prev[i, :2]
            yaw = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-9 else 0.0
            artists.extend(
                draw_quadrotor_2d(
                    ax_top,
                    p[i, :2],
                    yaw,
                    size=6.0,
                    arm_colors=(c_rgb[i], c_rgb[i]),
                    motor_color=c_rgb[i],
                    arm_lw=1.2,
                    motor_size=9,
                )
            )

        c_on.set_data(times[:step], cov_on[:step] * 100)
        c_naive.set_data(times[:step], naive_off[:step] * 100)
        c_off.set_data(times[:step], cov_off[:step] * 100)
        r_on.set_data(times[:step], reach_on[:step])
        r_off.set_data(times[:step], reach_off[:step])
        title.set_text(
            f"t = {step * DT:.0f} s   coverage {cov_on[step] * 100:.1f}%   "
            f"reachable {int(reach_on[step])}/{N_AGENTS}"
        )

    anim.animate(update, len(idx))
    anim.save()

    print(
        f"  relay       coverage {cov_on[-1] * 100:5.1f}%"
        f"   reachable {int(reach_on[-1])}/{N_AGENTS}"
    )
    print(
        f"  untethered  coverage {cov_off[-1] * 100:5.1f}%"
        f"   reachable {int(reach_off[-1])}/{N_AGENTS}"
        f"   (naive {naive_off[-1] * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
