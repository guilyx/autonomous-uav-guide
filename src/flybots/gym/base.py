# Erwin Lejeune - 2026-02-18
"""Base class for reinforcement-learning environments.

Follows the Gymnasium API — ``reset()`` returns ``(obs, info)`` and
``step()`` returns ``(obs, reward, terminated, truncated, info)`` — without
depending on Gymnasium being installed.

The ``terminated`` / ``truncated`` split matters for correct value
bootstrapping: ``terminated`` means the episode genuinely ended (the
aircraft crashed, the task was solved), while ``truncated`` means the time
limit ran out and the value function should still bootstrap from the final
state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flybots.gym.spaces import Box

__all__ = ["UAVEnv", "EnvConfig", "StepResult"]


@dataclass
class EnvConfig:
    """Settings shared by every environment in this package."""

    dt: float = 0.02
    """Control period [s]. The physics substeps below this."""
    max_episode_seconds: float = 10.0
    """Time limit before the episode is truncated."""
    physics_substeps: int = 2
    """Integration substeps per control step, for numerical stability."""
    seed: int | None = None

    action_smoothing_penalty: float = 0.01
    """Cost on changes in action, which discourages motor-shredding chatter."""
    action_magnitude_penalty: float = 0.005
    """Cost on control effort."""

    @property
    def max_episode_steps(self) -> int:
        return max(1, int(round(self.max_episode_seconds / self.dt)))

    @property
    def physics_dt(self) -> float:
        return self.dt / self.physics_substeps


@dataclass
class StepResult:
    """Structured form of the Gymnasium 5-tuple, for readability in envs."""

    observation: NDArray[np.floating]
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self):
        return (
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
        )


class UAVEnv(ABC):
    """Abstract UAV reinforcement-learning environment.

    Subclasses implement :meth:`_reset_task`, :meth:`_observe`,
    :meth:`_reward` and :meth:`_terminated`; this class owns the episode
    bookkeeping, action clipping, substepped integration and seeding.

    Examples
    --------
    >>> from flybots.gym import make
    >>> env = make("hover", seed=0)
    >>> obs, info = env.reset()
    >>> obs.shape == env.observation_space.shape
    True
    >>> action = env.action_space.sample(env.rng)
    >>> obs, reward, terminated, truncated, info = env.step(action)
    >>> isinstance(reward, float)
    True
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0
        self._previous_action: NDArray[np.floating] | None = None
        self._trajectory: list[NDArray[np.floating]] = []

    # ── spaces ────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def observation_space(self) -> Box: ...

    @property
    @abstractmethod
    def action_space(self) -> Box: ...

    # ── task hooks ────────────────────────────────────────────────────────

    @abstractmethod
    def _reset_task(self) -> None:
        """Randomise the vehicle and goal for a new episode."""

    @abstractmethod
    def _observe(self) -> NDArray[np.floating]:
        """Build the observation vector from the current state."""

    @abstractmethod
    def _apply_action(self, action: NDArray[np.floating]) -> None:
        """Advance the vehicle by one control period."""

    @abstractmethod
    def _reward(self, action: NDArray[np.floating]) -> tuple[float, dict[str, float]]:
        """Return ``(reward, breakdown)`` for the step just taken."""

    @abstractmethod
    def _terminated(self) -> tuple[bool, str]:
        """Return ``(done, reason)``. Reason is ``""`` when not done."""

    @property
    @abstractmethod
    def vehicle_position(self) -> NDArray[np.floating]:
        """World-frame position, used for rendering and logging."""

    # ── episode API ───────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """Start a new episode. Returns ``(observation, info)``."""
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self._previous_action = None
        self._reset_task()
        self._trajectory = [self.vehicle_position.copy()]
        return self._observe(), {"task": type(self).__name__}

    def step(self, action: NDArray[np.floating]):
        """Advance one control period. Returns the Gymnasium 5-tuple."""
        action = self.action_space.clip(np.asarray(action, dtype=np.float64).reshape(-1))
        self._apply_action(action)
        self.step_count += 1
        self._trajectory.append(self.vehicle_position.copy())

        reward, breakdown = self._reward(action)
        reward += self._action_cost(action)
        self._previous_action = action.copy()

        terminated, reason = self._terminated()
        truncated = (not terminated) and self.step_count >= self.config.max_episode_steps

        info: dict[str, Any] = {
            "reward_breakdown": breakdown,
            "steps": self.step_count,
            "time": self.step_count * self.config.dt,
        }
        if terminated:
            info["termination_reason"] = reason
        return StepResult(self._observe(), float(reward), terminated, truncated, info).as_tuple()

    def _action_cost(self, action: NDArray[np.floating]) -> float:
        """Penalise both control effort and rapid changes in it."""
        config = self.config
        cost = config.action_magnitude_penalty * float(np.sum(np.square(action)))
        if self._previous_action is not None:
            delta = action - self._previous_action
            cost += config.action_smoothing_penalty * float(np.sum(np.square(delta)))
        return -cost

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def trajectory(self) -> NDArray[np.floating]:
        """Positions visited this episode, shape ``(steps + 1, 3)``."""
        return np.array(self._trajectory)

    def seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def close(self) -> None:
        """Release resources. Nothing to do for these pure-NumPy envs."""

    @staticmethod
    def attitude_features(euler: NDArray[np.floating]) -> NDArray[np.floating]:
        """Encode Euler angles as sine/cosine pairs.

        Feeding raw angles to a policy puts a discontinuity at +-pi, where
        two numerically distant inputs describe the same attitude. The
        sin/cos encoding is continuous everywhere.
        """
        return np.concatenate([np.sin(euler), np.cos(euler)])
