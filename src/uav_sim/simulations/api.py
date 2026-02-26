"""Quad-first simulation composition API.

This API is additive and keeps existing run scripts compatible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment import default_world
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.simulations.mission_runner import run_standard_mission
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

from .contracts import (
    EnvironmentBundle,
    MissionConfig,
    PayloadPlugin,
    QuadPlatform,
    RunResult,
    SimulationConfig,
)


def create_environment(
    *,
    n_buildings: int = 6,
    world_size: float = 30.0,
    seed: int = 42,
) -> EnvironmentBundle:
    """Create a world bundle for a simulation."""
    world, obstacles = default_world(world_size=world_size, n_buildings=n_buildings, seed=seed)
    return EnvironmentBundle(world=world, obstacles=obstacles)


def spawn_quad_platform(*, position: NDArray[np.floating] | None = None) -> QuadPlatform:
    """Create a quad platform with default low-level controller."""
    vehicle = Quadrotor()
    if position is not None:
        vehicle.reset(position=position)
    return QuadPlatform(vehicle=vehicle, controller=CascadedPIDController())


def create_mission(
    *,
    path: NDArray[np.floating],
    standard,
    fallback_policy: str | None = None,
) -> MissionConfig:
    """Build a mission config object for the runner."""
    return MissionConfig(path=path, standard=standard, fallback_policy=fallback_policy)


def create_sim(*, name: str, out_dir: Path, mission: MissionConfig) -> SimulationConfig:
    """Build a top-level simulation config."""
    return SimulationConfig(name=name, out_dir=out_dir, mission=mission)


def run_sim(
    *,
    sim: SimulationConfig,
    env: EnvironmentBundle,
    platform: QuadPlatform,
    payloads: list[PayloadPlugin] | None = None,
) -> RunResult:
    """Execute a composed simulation mission."""
    if len(sim.mission.path) == 0:
        raise ValueError("Mission path cannot be empty.")
    start = np.array([sim.mission.path[0, 0], sim.mission.path[0, 1], 0.0])
    platform.reset(start)
    mission = run_standard_mission(
        platform.vehicle,
        platform.controller,
        sim.mission.path,
        standard=sim.mission.standard,
        obstacles=env.obstacles,
        fallback_policy=sim.mission.fallback_policy,
    )
    packets = []
    for payload in payloads or []:
        payload.reset()
    for frame, state in enumerate(mission.states):
        t = frame * sim.mission.standard.dt
        for payload in payloads or []:
            packets.append(payload.step(t=t, frame=frame, state=state, world=env.world))
    return RunResult(mission=mission, payload_packets=packets)
