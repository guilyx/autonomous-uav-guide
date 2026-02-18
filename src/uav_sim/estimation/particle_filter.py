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

    Parameters:
        state_dim: Dimension of the state vector.
        num_particles: Number of particles.
        f: Process model ``f(x, u, dt) → x_next`` (per particle).
        likelihood: ``likelihood(z, x) → p(z|x)`` scalar.
        process_noise_std: Standard deviation of additive process noise.
    """

    def __init__(
        self,
        state_dim: int,
        num_particles: int,
        f: Callable[..., NDArray[np.floating]],
        likelihood: Callable[..., float],
        process_noise_std: float = 0.01,
    ) -> None:
        self.n = state_dim
        self.N = num_particles
        self.f = f
        self.likelihood = likelihood
        self.noise_std = process_noise_std

        self.particles = np.zeros((self.N, self.n))
        self.weights = np.ones(self.N) / self.N

    def reset(
        self,
        x0: NDArray[np.floating],
        spread: float = 0.1,
    ) -> None:
        """Initialise particles around x0 with Gaussian spread."""
        self.particles = np.random.default_rng().normal(loc=x0, scale=spread, size=(self.N, self.n))
        self.weights = np.ones(self.N) / self.N

    def predict(self, u: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Propagate particles through the process model with noise."""
        noise = np.random.default_rng().normal(0, self.noise_std, self.particles.shape)
        for i in range(self.N):
            self.particles[i] = self.f(self.particles[i], u, dt) + noise[i]
        return self.estimate

    def update(self, z: NDArray[np.floating]) -> NDArray[np.floating]:
        """Update particle weights based on measurement likelihood, then resample."""
        for i in range(self.N):
            self.weights[i] = self.likelihood(z, self.particles[i])

        # Normalise weights.
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.N) / self.N

        # Systematic resampling.
        self._resample()
        return self.estimate

    def _resample(self) -> None:
        """Systematic resampling to avoid particle degeneracy."""
        cumsum = np.cumsum(self.weights)
        rng = np.random.default_rng()
        u0 = rng.uniform(0, 1.0 / self.N)
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
