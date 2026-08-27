# Erwin Lejeune - 2026-02-18
"""Environment registry and optional Gymnasium integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from flybots.gym.base import UAVEnv

__all__ = ["make", "list_envs", "ENV_SPECS", "EnvSpec", "register_gymnasium"]


@dataclass(frozen=True)
class EnvSpec:
    """Metadata for one registered environment."""

    env_id: str
    factory: Callable[..., UAVEnv]
    description: str
    vehicle: str
    difficulty: str

    def __str__(self) -> str:
        return f"{self.env_id:16} [{self.difficulty:6}] {self.description}"


def _hover(**kwargs) -> UAVEnv:
    from flybots.gym.quadrotor_envs import HoverEnv

    return HoverEnv(**kwargs)


def _waypoint(**kwargs) -> UAVEnv:
    from flybots.gym.quadrotor_envs import WaypointEnv

    return WaypointEnv(**kwargs)


def _trajectory(**kwargs) -> UAVEnv:
    from flybots.gym.quadrotor_envs import TrajectoryEnv

    return TrajectoryEnv(**kwargs)


def _landing(**kwargs) -> UAVEnv:
    from flybots.gym.quadrotor_envs import LandingEnv

    return LandingEnv(**kwargs)


def _fw_cruise(**kwargs) -> UAVEnv:
    from flybots.gym.fixed_wing_envs import FixedWingCruiseEnv

    return FixedWingCruiseEnv(**kwargs)


def _fw_waypoint(**kwargs) -> UAVEnv:
    from flybots.gym.fixed_wing_envs import FixedWingWaypointEnv

    return FixedWingWaypointEnv(**kwargs)


ENV_SPECS: dict[str, EnvSpec] = {
    spec.env_id: spec
    for spec in [
        EnvSpec(
            "hover",
            _hover,
            "Hold a point in space from a randomised upset.",
            "quadrotor",
            "easy",
        ),
        EnvSpec(
            "waypoint",
            _waypoint,
            "Fly to a random waypoint and settle on it.",
            "quadrotor",
            "medium",
        ),
        EnvSpec(
            "trajectory",
            _trajectory,
            "Track a moving Lissajous reference.",
            "quadrotor",
            "hard",
        ),
        EnvSpec(
            "landing",
            _landing,
            "Descend and touch down gently on a pad.",
            "quadrotor",
            "medium",
        ),
        EnvSpec(
            "fw-cruise",
            _fw_cruise,
            "Hold altitude, airspeed and course on a fixed wing.",
            "fixed-wing",
            "medium",
        ),
        EnvSpec(
            "fw-waypoint",
            _fw_waypoint,
            "Fly a fixed-wing route without stalling.",
            "fixed-wing",
            "hard",
        ),
    ]
}


def list_envs() -> list[EnvSpec]:
    """Every registered environment, in registration order."""
    return list(ENV_SPECS.values())


def make(env_id: str, *, seed: int | None = None, **kwargs: Any) -> UAVEnv:
    """Create an environment by name.

    Parameters
    ----------
    env_id
        One of the keys in :data:`ENV_SPECS`.
    seed
        Seed for the environment's random number generator.
    **kwargs
        Forwarded to the environment constructor, e.g. ``config=...``.

    Examples
    --------
    >>> from flybots.gym import make
    >>> env = make("hover", seed=0)
    >>> env.action_space.shape
    (4,)
    """
    if env_id not in ENV_SPECS:
        available = ", ".join(sorted(ENV_SPECS))
        raise KeyError(f"Unknown environment {env_id!r}. Available: {available}")
    env = ENV_SPECS[env_id].factory(**kwargs)
    if seed is not None:
        env.seed(seed)
        env.config.seed = seed
    return env


def register_gymnasium() -> list[str]:
    """Register these environments with Gymnasium, if it is installed.

    Lets the tasks be used with any standard RL library::

        import gymnasium as gym
        import flybots.gym  # registers on import
        env = gym.make("flybots/Hover-v0")

    Returns the registered ids, or an empty list when Gymnasium is absent.
    """
    try:
        import gymnasium
        from gymnasium.envs.registration import register
    except ImportError:
        return []

    registered = []
    for spec in list_envs():
        # "fw-cruise" -> "FwCruise"
        name = "".join(part.capitalize() for part in spec.env_id.split("-"))
        gym_id = f"flybots/{name}-v0"
        if gym_id in gymnasium.registry:
            registered.append(gym_id)
            continue
        register(
            id=gym_id,
            entry_point=_adapter_class(),
            kwargs={"env_id": spec.env_id},
        )
        registered.append(gym_id)
    return registered


_ADAPTER_CLASS: type | None = None


def _adapter_class() -> type:
    """Build the adapter class, subclassing ``gymnasium.Env``.

    ``gymnasium.make`` type-checks the entry point against ``gymnasium.Env``
    and refuses anything else, so the adapter has to inherit from it -- but
    Gymnasium is an optional dependency, and importing it at module scope
    would make `flybots.gym` unimportable without it. The class is therefore
    built on first use, behind the same import guard as registration, and
    cached so every environment shares one type.
    """
    global _ADAPTER_CLASS
    if _ADAPTER_CLASS is not None:
        return _ADAPTER_CLASS

    import gymnasium

    class _GymnasiumAdapter(gymnasium.Env):
        """Wraps a :class:`UAVEnv` so Gymnasium sees real ``gymnasium.spaces``.

        The environments already follow the Gymnasium step/reset contract, so
        this only has to translate the space objects.
        """

        def __init__(self, env_id: str, **kwargs: Any) -> None:
            import gymnasium.spaces as gym_spaces

            self._env = make(env_id, **kwargs)
            self.observation_space = gym_spaces.Box(
                low=self._env.observation_space.low,
                high=self._env.observation_space.high,
                dtype=self._env.observation_space.dtype,
            )
            self.action_space = gym_spaces.Box(
                low=self._env.action_space.low,
                high=self._env.action_space.high,
                dtype=self._env.action_space.dtype,
            )
            self.metadata = self._env.metadata

        def _cast(self, obs):
            """Match the observation to the dtype the space advertises.

            The underlying environments declare ``float32`` spaces and return
            ``float64`` arrays. Gymnasium's passive checker flags that as
            "obs is not within the observation space", which is correct: a
            wrapper or replay buffer that trusts ``observation_space.dtype``
            gets a silent widening on every step.
            """
            return np.asarray(obs, dtype=self.observation_space.dtype)

        def reset(self, **kwargs):
            result = self._env.reset(**kwargs)
            if isinstance(result, tuple):
                obs, info = result
                return self._cast(obs), info
            return self._cast(result), {}

        def step(self, action):
            obs, reward, terminated, truncated, info = self._env.step(action)
            return self._cast(obs), float(reward), bool(terminated), bool(truncated), info

        def render(self):
            return None

        def close(self):
            self._env.close()

    _ADAPTER_CLASS = _GymnasiumAdapter
    return _ADAPTER_CLASS
