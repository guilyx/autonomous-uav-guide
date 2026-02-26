"""Quad-first contracts for composing reusable simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.world import World
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.simulations.mission_runner import MissionResult
from uav_sim.simulations.standards import SimulationStandard
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


@dataclass(frozen=True)
class EnvironmentBundle:
    """World and static obstacle list used by a simulation."""

    world: World
    obstacles: list[Any]


@dataclass(frozen=True)
class MissionConfig:
    """Mission-level settings independent from platform internals."""

    path: NDArray[np.floating]
    standard: SimulationStandard
    fallback_policy: str | None = None


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level simulation definition for simple composition."""

    name: str
    out_dir: Path
    mission: MissionConfig


@dataclass(frozen=True)
class SensorPacket:
    """Normalized packet emitted by payload/sensors each step."""

    t: float
    frame: int
    data: dict[str, Any]


class PayloadPlugin(Protocol):
    """Extensible payload plugin contract (sensors, tools, carried modules)."""

    def reset(self) -> None: ...

    def step(
        self, *, t: float, frame: int, state: NDArray[np.floating], world: World
    ) -> SensorPacket: ...


@dataclass(frozen=True)
class RunResult:
    """Execution result with mission output and payload data."""

    mission: MissionResult
    payload_packets: list[SensorPacket]


@dataclass
class QuadPlatform:
    """Quadrotor platform with a standard low-level controller."""

    vehicle: Quadrotor
    controller: CascadedPIDController

    @property
    def state(self) -> NDArray[np.floating]:
        return self.vehicle.state

    def reset(self, position: NDArray[np.floating]) -> None:
        self.vehicle.reset(position=position)
