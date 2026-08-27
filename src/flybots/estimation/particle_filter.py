# Erwin Lejeune - 2026-02-16
"""Particle filter (sequential importance resampling) for state estimation.

Reference: M. S. Arulampalam et al., "A Tutorial on Particle Filters for
Online Nonlinear/Non-Gaussian Bayesian Tracking," IEEE TSP, 2002.
DOI: 10.1109/78.978374
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class ParticleFilter:
    """Sequential Importance Resampling (SIR) particle filter.

    Weights are accumulated multiplicatively and the particle set is
    resampled only when the effective sample size drops below
    ``resample_threshold · N`` (Arulampalam §III-D).  Resampling on every
    step, as a naive SIR does, throws away information and injects
    avoidable jitter into the estimate.

    Parameters:
        state_dim: Dimension of the state vector.
        num_particles: Number of particles.
        f: Process model ``f(x, u, dt) → x_next`` (per particle).
        likelihood: ``likelihood(z, x) → p(z|x)`` scalar.
        process_noise_std: Standard deviation of additive process noise.
            Scalar, or per-state array of length ``state_dim``.
        resample_threshold: Resample when ``N_eff < threshold · N``.
        seed: RNG seed.  Fixed by default so a run is reproducible —
            drawing from a fresh unseeded generator on every call, as an
            earlier version did, makes results irreproducible even when
            the surrounding simulation is seeded.
    """

    def __init__(
        self,
        state_dim: int,
        num_particles: int,
        f: Callable[..., NDArray[np.floating]],
        likelihood: Callable[..., float],
        process_noise_std: float | NDArray[np.floating] = 0.01,
        resample_threshold: float = 0.5,
        seed: int | None = 0,
    ) -> None:
        self.n = state_dim
        self.N = num_particles
        self.f = f
        self.likelihood = likelihood
        self.noise_std = np.broadcast_to(
            np.asarray(process_noise_std, dtype=float), (state_dim,)
        ).copy()
        self.resample_threshold = resample_threshold
        self._rng = np.random.default_rng(seed)

        self.particles = np.zeros((self.N, self.n))
        self.weights = np.ones(self.N) / self.N

    def reset(
        self,
        x0: NDArray[np.floating],
        spread: float | NDArray[np.floating] = 0.1,
    ) -> None:
        """Initialise particles around x0 with Gaussian spread."""
        self.particles = self._rng.normal(loc=x0, scale=spread, size=(self.N, self.n))
        self.weights = np.ones(self.N) / self.N

    @property
    def effective_sample_size(self) -> float:
        """``1 / Σ wᵢ²`` — how many particles are actually carrying weight."""
        return float(1.0 / np.sum(self.weights**2))

    def predict(self, u: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Propagate particles through the process model with noise."""
        noise = self._rng.normal(0.0, 1.0, self.particles.shape) * self.noise_std
        for i in range(self.N):
            self.particles[i] = self.f(self.particles[i], u, dt) + noise[i]
        return self.estimate

    def update(self, z: NDArray[np.floating]) -> NDArray[np.floating]:
        """Reweight particles by measurement likelihood, resampling if degenerate."""
        for i in range(self.N):
            self.weights[i] *= self.likelihood(z, self.particles[i])

        total = np.sum(self.weights)
        if total > 0 and np.isfinite(total):
            self.weights /= total
        else:
            # Every particle was ruled out: fall back to a uniform prior
            # rather than propagating NaNs through the estimate.
            self.weights = np.ones(self.N) / self.N

        if self.effective_sample_size < self.resample_threshold * self.N:
            self._resample()
        return self.estimate

    def _resample(self) -> None:
        """Systematic resampling to avoid particle degeneracy."""
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # guard against round-off leaving the last edge < 1
        u0 = self._rng.uniform(0, 1.0 / self.N)
        positions = u0 + np.arange(self.N) / self.N

        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    @property
    def estimate(self) -> NDArray[np.floating]:
        """Weighted mean of particles."""
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def variance(self) -> NDArray[np.floating]:
        """Weighted variance of particles."""
        mean = self.estimate
        diff = self.particles - mean
        return np.average(diff**2, weights=self.weights, axis=0)
