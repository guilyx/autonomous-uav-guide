# Erwin Lejeune - 2026-02-16
"""Model Predictive Path Integral (MPPI) sampling-based trajectory tracker.

Reference: G. Williams et al., "Information Theoretic MPC for Model-Based
Reinforcement Learning," ICRA, 2017. DOI: 10.1109/ICRA.2017.7989202
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class MPPITracker:
    """MPPI: derivative-free sampling-based trajectory optimiser.

    At each step, samples K random control sequences of length H,
    evaluates their cost, and computes a weighted average to obtain
    the optimal control.

    Parameters:
        state_dim: Dimension of state.
        control_dim: Dimension of control input.
        horizon: Number of time steps in the prediction horizon.
        num_samples: Number of random samples (K).
        lambda_: Temperature parameter (inverse of information cost weight).
        dynamics: ``f(x, u, dt) → x_next``.
        cost_fn: ``cost(x, u, ref) → float``.
        control_std: Standard deviation of control noise per dimension.
        dt: Time step for forward simulation.
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        horizon: int = 20,
        num_samples: int = 200,
        lambda_: float = 1.0,
        dynamics: Callable[..., NDArray[np.floating]] | None = None,
        cost_fn: Callable[..., float] | None = None,
        control_std: float | NDArray[np.floating] = 1.0,
        dt: float = 0.01,
    ) -> None:
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.K = num_samples
        self.lambda_ = lambda_
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.control_std = np.atleast_1d(np.asarray(control_std, dtype=np.float64))
        self.dt = dt

        # Warm-start: zero control sequence.
        self.U = np.zeros((horizon, control_dim))

    def reset(self) -> None:
        self.U = np.zeros((self.horizon, self.control_dim))

    def compute(
        self,
        state: NDArray[np.floating],
        reference: NDArray[np.floating] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.floating]:
        """Compute optimal control for the current state.

        Args:
            state: Current state vector.
            reference: Reference trajectory or target (passed to cost_fn).
            seed: Random seed for reproducibility.

        Returns:
            Control input for the current time step.
        """
        if self.dynamics is None or self.cost_fn is None:
            raise RuntimeError("dynamics and cost_fn must be set before calling compute.")

        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1, (self.K, self.horizon, self.control_dim)) * self.control_std
        costs = np.zeros(self.K)

        for k in range(self.K):
            x = state.copy()
            sample_cost = 0.0
            for t in range(self.horizon):
                u = self.U[t] + noise[k, t]
                sample_cost += self.cost_fn(x, u, reference)
                x = self.dynamics(x, u, self.dt)
            costs[k] = sample_cost

        # Compute weights.
        min_cost = np.min(costs)
        weights = np.exp(-1.0 / self.lambda_ * (costs - min_cost))
        weights /= np.sum(weights) + 1e-10

        # Weighted average of noise perturbations.
        for t in range(self.horizon):
            self.U[t] += np.sum(weights[:, None] * noise[:, t, :], axis=0)

        # Extract first control, then shift sequence.
        u_opt = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1] = np.zeros(self.control_dim)

        return u_opt
