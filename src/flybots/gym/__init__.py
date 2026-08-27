# Erwin Lejeune - 2026-02-18
"""Reinforcement-learning environments for UAV control.

Six tasks — hover, waypoint, trajectory tracking, landing, and two
fixed-wing tasks — built on the same physics the rest of the library uses,
plus a dependency-free trainer so a fresh install can learn to fly::

    from flybots.gym import make, train

    result = train("hover", generations=40)
    print(result.best_return)

    env = make("hover", seed=0)
    obs, _ = env.reset()
    action = result.policy.act(obs)

The environments follow the Gymnasium API without requiring Gymnasium. If
it *is* installed, :func:`register_gymnasium` exposes them as
``flybots/Hover-v0`` and friends for use with any standard RL library.
"""

from flybots.gym.base import EnvConfig, StepResult, UAVEnv
from flybots.gym.fixed_wing_envs import (
    FixedWingCruiseEnv,
    FixedWingEnvConfig,
    FixedWingWaypointEnv,
)
from flybots.gym.optimizers import (
    AugmentedRandomSearch,
    CrossEntropyMethod,
    build_optimizer,
)
from flybots.gym.policy import MLPPolicy, RunningNormalizer
from flybots.gym.quadrotor_envs import (
    HoverEnv,
    LandingEnv,
    QuadrotorEnvConfig,
    TrajectoryEnv,
    WaypointEnv,
)
from flybots.gym.registry import ENV_SPECS, EnvSpec, list_envs, make, register_gymnasium
from flybots.gym.spaces import Box
from flybots.gym.train import TrainConfig, TrainResult, evaluate, rollout, train

__all__ = [
    "ENV_SPECS",
    "AugmentedRandomSearch",
    "Box",
    "CrossEntropyMethod",
    "EnvConfig",
    "EnvSpec",
    "FixedWingCruiseEnv",
    "FixedWingEnvConfig",
    "FixedWingWaypointEnv",
    "HoverEnv",
    "LandingEnv",
    "MLPPolicy",
    "QuadrotorEnvConfig",
    "RunningNormalizer",
    "StepResult",
    "TrainConfig",
    "TrainResult",
    "TrajectoryEnv",
    "UAVEnv",
    "WaypointEnv",
    "build_optimizer",
    "evaluate",
    "list_envs",
    "make",
    "register_gymnasium",
    "rollout",
    "train",
]

# Registering on import is what makes `gym.make("flybots/Hover-v0")` work
# after a bare `import flybots.gym`. It is a no-op without Gymnasium.
register_gymnasium()
