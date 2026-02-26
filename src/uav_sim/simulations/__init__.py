# Erwin Lejeune - 2026-02-15
"""UAV simulation collection.

Each sub-package contains a ``run.py`` with a ``main()`` function that
generates a visualization GIF.  Run any simulation with::

    python -m uav_sim.simulations.<category>.<name>

Example::

    python -m uav_sim.simulations.path_tracking.pid_hover
"""

from .api import create_environment, create_mission, create_sim, run_sim, spawn_quad_platform
from .contracts import (
    EnvironmentBundle,
    MissionConfig,
    PayloadPlugin,
    QuadPlatform,
    RunResult,
    SimulationConfig,
)
from .plugins import PathPlannerPlugin, StaticPathPlanner

__all__ = [
    "EnvironmentBundle",
    "MissionConfig",
    "PayloadPlugin",
    "QuadPlatform",
    "RunResult",
    "SimulationConfig",
    "PathPlannerPlugin",
    "StaticPathPlanner",
    "create_environment",
    "spawn_quad_platform",
    "create_mission",
    "create_sim",
    "run_sim",
]
