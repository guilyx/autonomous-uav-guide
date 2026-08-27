# Erwin Lejeune - 2026-02-18
"""Derivative-free policy optimisers.

Two are provided, both pure NumPy:

:class:`AugmentedRandomSearch`
    Finite-difference gradient ascent with antithetic sampling and top-k
    direction selection. This is the default. Reference: H. Mania,
    A. Guy, B. Recht, "Simple random search provides a competitive approach
    to reinforcement learning", NeurIPS 2018.

:class:`CrossEntropyMethod`
    Fit a Gaussian to the best candidates and iterate. Reference:
    R. Y. Rubinstein, D. P. Kroese, *The Cross-Entropy Method*, 2004.

ARS is the default because it *estimates a direction* rather than a whole
distribution. CEM has to model a covariance over every parameter, so its
sample cost grows with dimension; ARS gets a usable gradient estimate from
a couple of dozen paired rollouts regardless of parameter count, which
matters when a policy has hundreds of weights and each evaluation costs a
full flight.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Optimizer",
    "AugmentedRandomSearch",
    "CrossEntropyMethod",
    "OptimizerStats",
    "build_optimizer",
]

ScoreFn = Callable[[NDArray[np.floating]], float]


@dataclass
class OptimizerStats:
    """What one optimiser iteration did."""

    iteration: int
    mean_return: float
    best_return: float
    elite_return: float
    step_size: float
    evaluations: int


class Optimizer(ABC):
    """Common interface: propose parameters, score them, update."""

    def __init__(self, parameters: NDArray[np.floating], seed: int | None = None) -> None:
        self.parameters = np.asarray(parameters, dtype=np.float64).copy()
        self.rng = np.random.default_rng(seed)
        self.iteration = 0

    @abstractmethod
    def step(self, score: ScoreFn) -> OptimizerStats:
        """Run one iteration, calling ``score`` on candidate parameters."""

    @property
    @abstractmethod
    def evaluations_per_step(self) -> int:
        """How many policy evaluations one :meth:`step` costs."""


class AugmentedRandomSearch(Optimizer):
    """ARS-V1t: antithetic finite differences with top-k direction selection.

    Each iteration samples ``directions`` random perturbations, scores the
    policy at plus and minus each one, keeps the ``top_directions`` whose
    pair produced the largest response, and steps along their
    return-weighted average.

    Dividing the step by the standard deviation of the surviving returns is
    what makes one learning rate work across tasks whose returns differ by
    orders of magnitude — without it, ``step_size`` has to be retuned for
    every reward scale.
    """

    def __init__(
        self,
        parameters: NDArray[np.floating],
        *,
        step_size: float = 0.02,
        noise: float = 0.03,
        directions: int = 16,
        top_directions: int | None = None,
        step_decay: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(parameters, seed)
        self.step_size = step_size
        self.noise = noise
        self.directions = directions
        self.top_directions = min(top_directions or max(1, directions // 2), directions)
        self.step_decay = step_decay

    @property
    def evaluations_per_step(self) -> int:
        return 2 * self.directions

    def step(self, score: ScoreFn) -> OptimizerStats:
        deltas = self.rng.normal(0.0, 1.0, size=(self.directions, self.parameters.size))
        plus = np.array([score(self.parameters + self.noise * d) for d in deltas])
        minus = np.array([score(self.parameters - self.noise * d) for d in deltas])

        # Rank directions by their strongest response in either sign, so a
        # direction only survives if perturbing along it actually matters.
        strength = np.maximum(plus, minus)
        chosen = np.argsort(strength)[-self.top_directions :]

        rewards = np.concatenate([plus[chosen], minus[chosen]])
        spread = float(rewards.std())
        if spread < 1e-8:
            # Every direction scored identically: no gradient information,
            # so widen the search instead of dividing by ~zero.
            self.noise *= 1.5
            self.iteration += 1
            return OptimizerStats(
                iteration=self.iteration,
                mean_return=float(rewards.mean()),
                best_return=float(strength.max()),
                elite_return=float(strength[chosen].mean()),
                step_size=self.step_size,
                evaluations=self.evaluations_per_step,
            )

        gradient = ((plus[chosen] - minus[chosen])[:, None] * deltas[chosen]).sum(axis=0)
        self.parameters += (self.step_size / (self.top_directions * spread)) * gradient
        self.step_size *= self.step_decay
        self.iteration += 1

        return OptimizerStats(
            iteration=self.iteration,
            mean_return=float(np.concatenate([plus, minus]).mean()),
            best_return=float(strength.max()),
            elite_return=float(strength[chosen].mean()),
            step_size=self.step_size,
            evaluations=self.evaluations_per_step,
        )


class CrossEntropyMethod(Optimizer):
    """Fit a diagonal Gaussian to the elite candidates and iterate.

    Simple and robust in low dimensions. The variance floor is the standard
    "CEM with noise" fix (Szita & Lorincz, 2006) that stops the population
    collapsing onto the first decent solution it finds.
    """

    def __init__(
        self,
        parameters: NDArray[np.floating],
        *,
        population: int = 40,
        elite_fraction: float = 0.2,
        initial_std: float = 0.15,
        std_floor: float = 0.02,
        std_decay: float = 0.97,
        seed: int | None = None,
    ) -> None:
        super().__init__(parameters, seed)
        self.population = population
        self.elite_count = max(2, int(round(population * elite_fraction)))
        self.std = np.full(self.parameters.size, initial_std)
        self.std_floor = std_floor
        self.std_decay = std_decay

    @property
    def evaluations_per_step(self) -> int:
        return self.population

    def step(self, score: ScoreFn) -> OptimizerStats:
        candidates = self.rng.normal(
            self.parameters, self.std, size=(self.population, self.parameters.size)
        )
        scores = np.array([score(theta) for theta in candidates])
        elite_indices = np.argsort(scores)[-self.elite_count :]
        elite = candidates[elite_indices]

        self.parameters = elite.mean(axis=0)
        self.std = np.maximum(elite.std(axis=0), self.std_floor) * self.std_decay
        self.iteration += 1

        return OptimizerStats(
            iteration=self.iteration,
            mean_return=float(scores.mean()),
            best_return=float(scores.max()),
            elite_return=float(scores[elite_indices].mean()),
            step_size=float(self.std.mean()),
            evaluations=self.evaluations_per_step,
        )


def build_optimizer(
    name: str,
    parameters: NDArray[np.floating],
    *,
    seed: int | None = None,
    **kwargs,
) -> Optimizer:
    """Create an optimiser by name (``"ars"`` or ``"cem"``)."""
    normalised = name.lower().replace("-", "").replace("_", "")
    if normalised in {"ars", "augmentedrandomsearch"}:
        return AugmentedRandomSearch(parameters, seed=seed, **kwargs)
    if normalised in {"cem", "crossentropymethod"}:
        return CrossEntropyMethod(parameters, seed=seed, **kwargs)
    raise ValueError(f"Unknown optimizer {name!r}. Use 'ars' or 'cem'.")
