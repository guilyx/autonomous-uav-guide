# Erwin Lejeune - 2026-02-18
"""Reinforcement-learning environments for UAV control.

Six tasks — hover, waypoint, trajectory tracking, landing, and two
fixed-wing tasks — built on the same physics the rest of the library uses,
plus a dependency-free trainer so a fresh install can learn to fly::

    from uav_sim.gym import make, train

    result = train("hover", generations=40)
    print(result.best_return)

    env = make("hover", seed=0)
    obs, _ = env.reset()
    action = result.policy.act(obs)

The environments follow the Gymnasium API without requiring Gymnasium. If
it *is* installed, :func:`register_gymnasium` exposes them as
``uav_sim/Hover-v0`` and friends for use with any standard RL library.
"""

from uav_sim.gym.base import EnvConfig, StepResult, UAVEnv
from uav_sim.gym.fixed_wing_envs import (
    FixedWingCruiseEnv,
    FixedWingEnvConfig,
    FixedWingWaypointEnv,
)
from uav_sim.gym.optimizers import (
    AugmentedRandomSearch,
    CrossEntropyMethod,
    build_optimizer,
)
from uav_sim.gym.policy import MLPPolicy, RunningNormalizer
from uav_sim.gym.quadrotor_envs import (
    HoverEnv,
    LandingEnv,
    QuadrotorEnvConfig,
    TrajectoryEnv,
    WaypointEnv,
)
from uav_sim.gym.registry import ENV_SPECS, EnvSpec, list_envs, make, register_gymnasium
from uav_sim.gym.spaces import Box
from uav_sim.gym.train import TrainConfig, TrainResult, evaluate, rollout, train

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

# Registering on import is what makes `gym.make("uav_sim/Hover-v0")` work
# after a bare `import uav_sim.gym`. It is a no-op without Gymnasium.
register_gymnasium()
