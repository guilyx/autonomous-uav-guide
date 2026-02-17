# Erwin Lejeune - 2026-02-17
"""Particle filter: live particle cloud + weight distribution visualisation.

Shows the particle cloud evolving in real-time with colour-coded weights,
alongside the true/estimated position and a weight histogram.

Reference: M. S. Arulampalam et al., "A Tutorial on Particle Filters," IEEE
TSP, 2002. DOI: 10.1109/78.978374
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.estimation.particle_filter import ParticleFilter
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def _f_single(x, _u, dt):
    return np.array([x[0] + x[1] * dt, x[1]])


def _likelihood(z, x):
    return float(np.exp(-0.5 * (x[0] - z[0]) ** 2 / 0.5))


def main() -> None:
    dt, N = 0.05, 200
    rng = np.random.default_rng(42)
    pf = ParticleFilter(
        state_dim=2,
        num_particles=300,
        f=_f_single,
        likelihood=_likelihood,
        process_noise_std=0.3,
    )
    pf.reset(np.array([0.0, 1.0]), spread=1.0)
    xt = np.array([0.0, 1.0])
    true_s = np.zeros((N, 2))
    est_s = np.zeros((N, 2))
    meas = np.zeros(N)
    part_hist: list[np.ndarray] = []
    weight_hist: list[np.ndarray] = []

    for i in range(N):
        xt = np.array([xt[0] + xt[1] * dt, xt[1]]) + rng.normal(0, 0.02, 2)
        z = np.array([xt[0] + rng.normal(0, 0.7)])
        pf.predict(np.zeros(1), dt)
        pf.update(z)
        true_s[i] = xt
        est_s[i] = pf.estimate
        meas[i] = z[0]
        part_hist.append(pf.particles.copy())
        weight_hist.append(pf.weights.copy())

    times = np.arange(N) * dt
    skip = max(1, N // 150)
    idx = list(range(0, N, skip))
    n_frames = len(idx)

    anim = SimAnimator("particle_filter", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(13, 7))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.35, wspace=0.3)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_neff = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])
    fig.suptitle("Particle Filter — Live Simulation", fontsize=13)

    ylo = min(true_s[:, 0].min(), est_s[:, 0].min()) - 2
    yhi = max(true_s[:, 0].max(), est_s[:, 0].max()) + 2
    ax_main.set_xlim(0, N * dt)
    ax_main.set_ylim(ylo, yhi)
    ax_main.set_xlabel("Time [s]")
    ax_main.set_ylabel("Position")
    ax_main.grid(True, alpha=0.2)
    (lt,) = ax_main.plot([], [], "k-", lw=1.5, label="True")
    (le,) = ax_main.plot([], [], "b-", lw=1.2, label="PF est")
    (sm,) = ax_main.plot([], [], "r.", ms=3, alpha=0.4, label="Meas")
    cloud = ax_main.scatter([], [], s=2, c=[], cmap="cool", alpha=0.4, vmin=0, vmax=1)
    ax_main.legend(fontsize=7, loc="upper left")

    n_eff = np.array([1.0 / np.sum(w**2) for w in weight_hist])
    ax_neff.set_xlim(0, N * dt)
    ax_neff.set_ylim(0, 350)
    ax_neff.set_ylabel("N_eff", fontsize=9)
    ax_neff.grid(True, alpha=0.3)
    ax_neff.axhline(150, color="r", ls="--", lw=0.8, alpha=0.5, label="Resample thresh")
    (lneff,) = ax_neff.plot([], [], "b-", lw=1)
    ax_neff.legend(fontsize=6, loc="upper right")
    ax_neff.tick_params(labelsize=7)

    err = np.abs(true_s[:, 0] - est_s[:, 0])
    ax_err.set_xlim(0, N * dt)
    ax_err.set_ylim(0, max(1.0, err.max() * 1.1))
    ax_err.set_xlabel("Time [s]", fontsize=9)
    ax_err.set_ylabel("|Error|", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    (lerr,) = ax_err.plot([], [], "r-", lw=1)
    ax_err.tick_params(labelsize=7)

    def update(f):
        k = idx[f]
        lt.set_data(times[:k], true_s[:k, 0])
        le.set_data(times[:k], est_s[:k, 0])
        sm.set_data(times[:k], meas[:k])

        p = part_hist[k]
        w = weight_hist[k]
        w_norm = w / (w.max() + 1e-12)
        offsets = np.column_stack([np.full(len(p), times[k]), p[:, 0]])
        cloud.set_offsets(offsets)
        cloud.set_array(w_norm)

        lneff.set_data(times[:k], n_eff[:k])
        lerr.set_data(times[:k], err[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
