# Erwin Lejeune - 2026-02-15
"""Particle filter tracking with visible particle cloud.

Reference: M. S. Arulampalam et al., "A Tutorial on Particle Filters," IEEE
TSP, 2002. DOI: 10.1109/78.978374
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.estimation.particle_filter import ParticleFilter
from quadrotor_sim.visualization import SimAnimator


def _f_single(x, _u, dt):
    return np.array([x[0] + x[1] * dt, x[1]])


def _likelihood(z, x):
    return float(np.exp(-0.5 * (x[0] - z[0]) ** 2 / 0.5))


def main() -> None:
    dt, N = 0.05, 200
    rng = np.random.default_rng(42)
    pf = ParticleFilter(
        state_dim=2, num_particles=300, f=_f_single, likelihood=_likelihood, process_noise_std=0.3
    )
    pf.reset(np.array([0.0, 1.0]), spread=1.0)
    xt = np.array([0.0, 1.0])
    true_s = np.zeros((N, 2))
    est_s = np.zeros((N, 2))
    meas = np.zeros(N)
    part_hist: list[np.ndarray] = []
    for i in range(N):
        xt = np.array([xt[0] + xt[1] * dt, xt[1]]) + rng.normal(0, 0.02, 2)
        z = np.array([xt[0] + rng.normal(0, 0.7)])
        pf.predict(np.zeros(1), dt)
        pf.update(z)
        true_s[i] = xt
        est_s[i] = pf.estimate
        meas[i] = z[0]
        part_hist.append(pf.particles.copy())
    times = np.arange(N) * dt
    skip = max(1, N // 150)
    idx = list(range(0, N, skip))
    anim = SimAnimator("particle_filter", out_dir=Path(__file__).parent)
    _, axes = anim.figure_2d("Particle Filter", nrows=1)
    ax = axes[0]
    ylo = min(true_s[:, 0].min(), est_s[:, 0].min()) - 2
    yhi = max(true_s[:, 0].max(), est_s[:, 0].max()) + 2
    ax.set_xlim(0, N * dt)
    ax.set_ylim(ylo, yhi)
    (lt,) = ax.plot([], [], "k-", lw=1.5, label="True")
    (le,) = ax.plot([], [], "b-", lw=1.2, label="PF est")
    (sm,) = ax.plot([], [], "r.", ms=3, alpha=0.4, label="Meas")
    cloud = ax.scatter([], [], s=1, c="cyan", alpha=0.15, label="Particles")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    def update(f):
        k = idx[f]
        lt.set_data(times[:k], true_s[:k, 0])
        le.set_data(times[:k], est_s[:k, 0])
        sm.set_data(times[:k], meas[:k])
        p = part_hist[k]
        cloud.set_offsets(np.column_stack([np.full(len(p), times[k]), p[:, 0]]))

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
