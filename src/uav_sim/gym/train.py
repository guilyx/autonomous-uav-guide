# Erwin Lejeune - 2026-02-18
"""Trainer for the UAV environments.

Derivative-free policy search, pure NumPy. Training a drone to fly should
not require a deep-learning stack — these tasks are low-dimensional enough
that a linear policy driven by random search solves them, and being able to
run ``flybots train hover`` on a bare ``pip install`` matters more here than
the last few percent of return.

The optimiser is pluggable; see :mod:`uav_sim.gym.optimizers`. The default
is Augmented Random Search.

    >>> from uav_sim.gym.train import train
    >>> result = train("hover", iterations=2, directions=4, seed=0)
    >>> result.best_return > -1e9
    True
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from uav_sim.gym.optimizers import build_optimizer
from uav_sim.gym.policy import MLPPolicy
from uav_sim.gym.registry import make

__all__ = ["TrainConfig", "TrainResult", "train", "evaluate", "rollout"]


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    optimizer: str = "ars"
    """``"ars"`` (default) or ``"cem"``. See :mod:`uav_sim.gym.optimizers`."""
    iterations: int = 120
    directions: int = 16
    """ARS: perturbation directions per iteration (each costs two episodes)."""
    top_directions: int = 8
    """ARS: how many of them actually contribute to the update."""
    population: int = 40
    """CEM only: candidates per generation."""

    step_size: float = 0.02
    noise: float = 0.03
    """Finite-difference perturbation size in parameter space."""

    episodes_per_candidate: int = 8
    """Episodes averaged per evaluation. More cuts noise, costs time."""
    fixed_task_seeds: bool = True
    """Score every candidate on the *same* episodes for the whole run.

    This matters more than any other knob here. These environments
    randomise the spawn heavily, so with fresh seeds each iteration the gap
    between two candidates' scores is dominated by which starts they
    happened to draw — finite differences over that measure luck, not
    gradient. Holding the seed set fixed makes the objective a
    deterministic function of the parameters, which is what random search
    needs. Generalisation is then checked separately on unseen seeds."""

    hidden_sizes: tuple[int, ...] = ()
    """Policy hidden layers. Empty means a linear policy.

    Linear is the default deliberately: on body-frame errors a linear
    policy *is* structurally a PD controller, which is the shape of the
    solution these tasks want, and it needs ~80 parameters instead of
    several thousand. See Mania et al., "Simple random search provides a
    competitive approach to reinforcement learning", 2018.
    """
    normalize_observations: bool = False
    """Whiten observations. Off by default; see :class:`MLPPolicy`."""
    calibration_episodes: int = 10
    """Episodes used to fit the observation normaliser, when enabled."""
    evaluate_every: int = 5
    """How often to score the current policy on held-out seeds."""
    seed: int | None = None


@dataclass
class TrainResult:
    """Outcome of a training run."""

    policy: MLPPolicy
    best_return: float
    history: list[dict[str, float]] = field(default_factory=list)
    env_id: str = ""
    elapsed_seconds: float = 0.0
    total_episodes: int = 0

    @property
    def returns(self) -> NDArray[np.floating]:
        return np.array([row["mean_return"] for row in self.history])

    @property
    def best_returns(self) -> NDArray[np.floating]:
        return np.array([row["best_return"] for row in self.history])


def rollout(
    env,
    policy: MLPPolicy,
    parameters: NDArray[np.floating] | None = None,
    *,
    seed: int | None = None,
    collect_observations: bool = False,
) -> tuple[float, int, dict]:
    """Run one episode. Returns ``(return, steps, info)``."""
    observation, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    observations = [observation] if collect_observations else None
    last_info: dict = {}

    while True:
        action = policy.act(observation, parameters)
        observation, reward, terminated, truncated, last_info = env.step(action)
        total += reward
        steps += 1
        if collect_observations:
            observations.append(observation)
        if terminated or truncated:
            break

    if collect_observations:
        last_info = {**last_info, "observations": np.array(observations)}
    return total, steps, last_info


def _calibrate_normalizer(env, policy: MLPPolicy, rng, episodes: int) -> None:
    """Seed the observation statistics before the search starts.

    Fitted once and then frozen. Refreshing it every iteration looks
    appealing but quietly makes the objective non-stationary: the same
    parameter vector means something different once the statistics move, so
    a policy carried forward is no longer the one that earned its score.
    """
    for _ in range(episodes):
        _, _, info = rollout(
            env,
            policy,
            # Perturb around the equilibrium policy rather than sampling
            # wildly, so the statistics describe states near actual flight.
            policy.parameters + rng.normal(0.0, 0.05, size=policy.parameter_count),
            seed=int(rng.integers(0, 2**31 - 1)),
            collect_observations=True,
        )
        policy.normalizer.update(info["observations"])


def _make_scorer(env, policy: MLPPolicy, episodes: int, seeds: list[int]):
    """Score parameters as the mean return over a fixed set of episode seeds.

    Holding the seeds fixed within an iteration is what makes the
    comparison meaningful: otherwise a mediocre policy that drew easy
    starts outranks a good one that drew hard ones, and the update follows
    luck instead of merit.
    """

    def score(parameters: NDArray[np.floating]) -> float:
        total = 0.0
        for seed in seeds[:episodes]:
            episode_return, _, _ = rollout(env, policy, parameters, seed=seed)
            total += episode_return
        return total / episodes

    return score


def train(
    env_id: str = "hover",
    *,
    iterations: int | None = None,
    directions: int | None = None,
    config: TrainConfig | None = None,
    seed: int | None = None,
    progress: Callable[[dict[str, float]], None] | None = None,
    save_path: str | Path | None = None,
) -> TrainResult:
    """Train a flight policy.

    Parameters
    ----------
    env_id
        Environment name, see :func:`uav_sim.gym.registry.list_envs`.
    iterations, directions
        Convenience overrides for the matching :class:`TrainConfig` fields.
    config
        Full hyperparameter set. Created with defaults when omitted.
    seed
        Seed for the search and the environment.
    progress
        Called once per iteration with a stats dict, for CLI output.
    save_path
        If given, the best policy is written here as ``.npz``.
    """
    config = config or TrainConfig()
    if iterations is not None:
        config.iterations = iterations
    if directions is not None:
        config.directions = directions
    if seed is not None:
        config.seed = seed

    rng = np.random.default_rng(config.seed)
    env = make(env_id, seed=config.seed)
    policy = MLPPolicy(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        hidden_sizes=config.hidden_sizes,
        seed=config.seed,
        normalize=config.normalize_observations,
    )
    if config.normalize_observations:
        _calibrate_normalizer(env, policy, rng, config.calibration_episodes)

    if config.optimizer.lower() == "cem":
        optimizer = build_optimizer(
            "cem", policy.parameters, seed=config.seed, population=config.population
        )
    else:
        optimizer = build_optimizer(
            "ars",
            policy.parameters,
            seed=config.seed,
            step_size=config.step_size,
            noise=config.noise,
            directions=config.directions,
            top_directions=config.top_directions,
        )

    best_parameters = optimizer.parameters.copy()
    best_return = -np.inf
    history: list[dict[str, float]] = []
    episodes_used = 0
    started = time.perf_counter()

    task_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(config.episodes_per_candidate)]

    for iteration in range(config.iterations):
        seeds = (
            task_seeds
            if config.fixed_task_seeds
            else [int(rng.integers(0, 2**31 - 1)) for _ in range(config.episodes_per_candidate)]
        )
        score = _make_scorer(env, policy, config.episodes_per_candidate, seeds)
        stats = optimizer.step(score)
        episodes_used += stats.evaluations * config.episodes_per_candidate

        # Periodically score the *current* policy on unseen seeds. Tracking
        # best-so-far by the highest perturbation score instead would keep
        # whichever candidate drew the kindest episodes, and the returned
        # policy would reliably evaluate worse than its training number.
        should_check = iteration % config.evaluate_every == 0 or iteration == config.iterations - 1
        if should_check:
            holdout = [int(rng.integers(0, 2**31 - 1)) for _ in range(4)]
            holdout_score = _make_scorer(env, policy, 4, holdout)(optimizer.parameters)
            episodes_used += 4
            if holdout_score > best_return:
                best_return = holdout_score
                best_parameters = optimizer.parameters.copy()
        else:
            holdout_score = float("nan")

        row = {
            "iteration": float(iteration),
            "mean_return": stats.mean_return,
            "best_return": stats.best_return,
            "elite_return": stats.elite_return,
            "holdout_return": holdout_score,
            "step_size": stats.step_size,
        }
        history.append(row)
        if progress is not None:
            progress(row)

    policy.parameters = best_parameters
    result = TrainResult(
        policy=policy,
        best_return=float(best_return),
        history=history,
        env_id=env_id,
        elapsed_seconds=time.perf_counter() - started,
        total_episodes=episodes_used,
    )
    if save_path is not None:
        policy.save(save_path)
    env.close()
    return result


def evaluate(
    env_id: str,
    policy: MLPPolicy,
    *,
    episodes: int = 20,
    seed: int | None = 0,
) -> dict[str, float]:
    """Score a trained policy over fresh episodes.

    Reports the share of episodes ending in each termination reason
    alongside the return, because *how* a policy fails is usually more
    informative than the number itself.
    """
    env = make(env_id, seed=seed)
    rng = np.random.default_rng(seed)
    returns, lengths, reasons = [], [], []

    for _ in range(episodes):
        episode_return, steps, info = rollout(env, policy, seed=int(rng.integers(0, 2**31 - 1)))
        returns.append(episode_return)
        lengths.append(steps)
        reasons.append(info.get("termination_reason", "time_limit"))

    env.close()
    summary = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes": float(episodes),
    }
    for reason in set(reasons):
        summary[f"pct_{reason}"] = 100.0 * reasons.count(reason) / episodes
    return summary
