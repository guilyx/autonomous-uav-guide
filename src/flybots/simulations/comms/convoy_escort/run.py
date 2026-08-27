# Erwin Lejeune - 2026-08-27
"""Convoy escort: a relay chain that grows as the convoy outruns its radio.

A ground vehicle drives a fixed route away from a base station. It carries
no long-range radio, so it stays reachable only while a chain of UAVs
bridges the gap -- and the gap grows, so the chain has to lengthen, then
shorten again as the route curves back.

Nothing here plans the chain. Each aircraft is pulled toward the midpoint
of the link it is responsible for and pushed by the shared connectivity
gradient; the number of hops the route needs is an outcome, not an input.

The comparison run keeps the escorts in a fixed formation around the
convoy, which is the obvious thing to do and loses the base entirely once
the convoy is far enough out.

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

from flybots.comms import GaussianLink, connectivity_gradient, hop_counts
from flybots.logging import SimLogger
from flybots.visualization import SimAnimator
from flybots.visualization.vehicle_artists import draw_quadrotor_2d

matplotlib.use("Agg")

WORLD = 420.0
ALT = 40.0
N_ESCORTS = 6
SIGMA = 42.0
LINK_THRESHOLD = 0.5
DT = 0.05
STEPS = 2600
BASE = np.array([60.0, 60.0, ALT])
CONVOY_SPEED = 7.0


def _convoy_position(t: float) -> np.ndarray:
    """A route that runs out and curves back, so the chain must grow then shrink."""
    s = CONVOY_SPEED * t
    return np.array(
        [
            60.0 + 300.0 * np.sin(s / 300.0),
            60.0 + 240.0 * np.sin(s / 150.0) ** 2,
            ALT,
        ]
    )


def _fly(relay: bool):
    link = GaussianLink(SIGMA)
    # Fleet layout: index 0 is the base, 1..N are escorts, last is the convoy.
    n = 1 + N_ESCORTS + 1
    pos = np.tile(BASE, (n, 1)).astype(float)
    pos[1:-1] += np.linspace(8.0, 40.0, N_ESCORTS)[:, None] * np.array([1.0, 0.6, 0.0])
    pos[-1] = _convoy_position(0.0)
    vel = np.zeros_like(pos)

    history = np.zeros((STEPS, n, 3))
    linked = np.zeros(STEPS)
    hops = np.zeros(STEPS)
    chain_span = np.zeros(STEPS)

    for step in range(STEPS):
        t = step * DT
        pos[0] = BASE
        pos[-1] = _convoy_position(t)

        escorts = slice(1, 1 + N_ESCORTS)
        if relay:
            # Each escort owns one link of the chain from base to convoy and
            # is drawn to its midpoint; the connectivity gradient then tidies
            # the spacing so no single hop is left overstretched.
            anchors = np.linspace(0.0, 1.0, N_ESCORTS + 2)[1:-1]
            targets = BASE + anchors[:, None] * (pos[-1] - BASE)
            command = 1.4 * (targets - pos[escorts]) - 2.2 * vel[escorts]
            command = command + 18.0 * connectivity_gradient(pos, link)[escorts]
        else:
            # The obvious alternative: sit on the convoy in formation.
            ring = np.linspace(0.0, 2.0 * np.pi, N_ESCORTS, endpoint=False)
            targets = pos[-1] + 22.0 * np.column_stack(
                [np.cos(ring), np.sin(ring), np.zeros(N_ESCORTS)]
            )
            command = 1.4 * (targets - pos[escorts]) - 2.2 * vel[escorts]

        command = np.clip(command, -6.0, 6.0)
        vel[escorts] = vel[escorts] + command * DT
        pos[escorts] = pos[escorts] + vel[escorts] * DT
        pos[:, 2] = ALT

        history[step] = pos
        h = hop_counts(link.weights(pos), 0, LINK_THRESHOLD)
        linked[step] = float(np.isfinite(h[-1]))
        hops[step] = float(h[-1]) if np.isfinite(h[-1]) else np.nan
        chain_span[step] = float(np.linalg.norm(pos[-1] - BASE))

    return history, linked, hops, chain_span


def main() -> None:
    relay, linked_on, hops_on, span = _fly(relay=True)
    _, linked_off, hops_off, _ = _fly(relay=False)
    times = np.arange(STEPS) * DT

    logger = SimLogger("convoy_escort", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "Relay chain escort")
    logger.log_metadata("n_escorts", N_ESCORTS)
    logger.log_metadata("sigma_m", SIGMA)
    for step in range(STEPS):
        logger.log_step(
            t=times[step],
            positions=relay[step],
            convoy_linked=bool(linked_on[step]),
            hops_to_convoy=float(hops_on[step]) if np.isfinite(hops_on[step]) else -1.0,
            convoy_range_m=float(span[step]),
        )
    logger.log_summary("linked_fraction_relay", float(np.mean(linked_on)))
    logger.log_summary("linked_fraction_formation", float(np.mean(linked_off)))
    logger.log_summary("max_hops", float(np.nanmax(hops_on)))
    logger.log_summary("max_convoy_range_m", float(span.max()))
    logger.save()

    skip = max(1, STEPS // 150)
    idx = list(range(0, STEPS, skip))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ax_top = fig.add_subplot(gs[:, 0])
    ax_link = fig.add_subplot(gs[0, 1])
    ax_hops = fig.add_subplot(gs[1, 1])

    fig.suptitle("Convoy Escort — the chain lengthens as the convoy pulls away", fontsize=13)

    ax_top.set_xlim(0, WORLD)
    ax_top.set_ylim(0, WORLD)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X [m]")
    ax_top.set_ylabel("Y [m]")
    ax_top.set_title("Base (square), escorts (dots), convoy (triangle)", fontsize=9)
    ax_top.grid(True, alpha=0.15)
    route = np.array([_convoy_position(s * DT) for s in range(STEPS)])
    ax_top.plot(route[:, 0], route[:, 1], color="0.75", lw=1.0, ls="--")

    ax_link.set_xlim(0, times[-1])
    ax_link.set_ylim(-0.1, 1.2)
    ax_link.set_xlabel("Time [s]", fontsize=8)
    ax_link.set_ylabel("Convoy reachable", fontsize=8)
    ax_link.set_title("Can the convoy reach the base?", fontsize=9)
    ax_link.grid(True, alpha=0.3)
    (k_on,) = ax_link.plot([], [], color="tab:green", lw=1.6, label="relay chain")
    (k_off,) = ax_link.plot([], [], color="tab:red", lw=1.3, label="fixed formation")
    ax_link.legend(fontsize=7, loc="center left")

    ax_hops.set_xlim(0, times[-1])
    ax_hops.set_ylim(0, N_ESCORTS + 2)
    ax_hops.set_xlabel("Time [s]", fontsize=8)
    ax_hops.set_ylabel("Hops to convoy", fontsize=8)
    ax_hops.set_title("Chain length is an outcome, not a setting", fontsize=9)
    ax_hops.grid(True, alpha=0.3)
    (h_on,) = ax_hops.plot([], [], color="tab:green", lw=1.6)
    ax_hops_r = ax_hops.twinx()
    ax_hops_r.set_ylabel("Convoy range from base [m]", fontsize=8, color="0.45")
    ax_hops_r.set_ylim(0, span.max() * 1.15)
    (r_line,) = ax_hops_r.plot([], [], color="0.55", lw=1.0, ls="--")

    artists: list = []
    link_model = GaussianLink(SIGMA)

    anim = SimAnimator("convoy_escort", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(frame: int) -> None:
        step = idx[frame]
        p = relay[step]

        for artist in artists:
            artist.remove()
        artists.clear()

        w = link_model.weights(p)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if w[i, j] > LINK_THRESHOLD:
                    (ln,) = ax_top.plot(
                        [p[i, 0], p[j, 0]],
                        [p[i, 1], p[j, 1]],
                        color="tab:cyan",
                        lw=1.0,
                        alpha=0.7,
                    )
                    artists.append(ln)

        (base,) = ax_top.plot(p[0, 0], p[0, 1], "s", color="white", ms=11, mec="k", mew=1.4)
        (conv,) = ax_top.plot(p[-1, 0], p[-1, 1], "^", color="tab:orange", ms=11, mec="k")
        artists.extend([base, conv])
        # The escorts are aircraft, so draw them as aircraft, pointed along
        # the leg of the chain they are flying.
        # Heading over several steps, not one: a single step at low
        # speed is mostly integrator wobble and the model spins.
        prev = relay[max(step - 12, 0)]
        for i in range(1, len(p) - 1):
            d = p[i, :2] - prev[i, :2]
            yaw = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-9 else 0.0
            artists.extend(
                draw_quadrotor_2d(
                    ax_top,
                    p[i, :2],
                    yaw,
                    size=9.0,
                    arm_colors=("tab:blue", "tab:blue"),
                    motor_color="tab:blue",
                    arm_lw=1.3,
                    motor_size=10,
                )
            )

        k_on.set_data(times[:step], linked_on[:step])
        k_off.set_data(times[:step], linked_off[:step])
        h_on.set_data(times[:step], hops_on[:step])
        r_line.set_data(times[:step], span[:step])

    anim.animate(update, len(idx))
    anim.save()

    print(f"  relay chain      convoy linked {np.mean(linked_on) * 100:5.1f}% of the run")
    print(f"  fixed formation  convoy linked {np.mean(linked_off) * 100:5.1f}% of the run")
    print(f"  max hops {np.nanmax(hops_on):.0f}   max convoy range {span.max():.0f} m")


if __name__ == "__main__":
    main()
