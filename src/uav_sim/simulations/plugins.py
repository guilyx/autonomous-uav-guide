"""Plugin protocols to make algorithms swappable in simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.world import World
from uav_sim.simulations.standards import SimulationStandard


class PathPlannerPlugin(Protocol):
    """Generate a trajectory/path for a mission from world + standard."""

    def plan(self, *, world: World, standard: SimulationStandard) -> NDArray[np.floating]: ...


class TrackerPlugin(Protocol):
    """Compute target commands from state and references."""

    def compute(
        self, state: NDArray[np.floating], reference: NDArray[np.floating]
    ) -> NDArray[np.floating]: ...


class EstimatorPlugin(Protocol):
    """State estimator contract used by benchmarks/sim loops."""

    def reset(self) -> None: ...

    def predict(self, u: NDArray[np.floating], dt: float) -> None: ...

    def update(self, z: NDArray[np.floating]) -> None: ...


class PerceptionPlugin(Protocol):
    """Perception feature extraction contract for sensor-driven pipelines."""

    def reset(self) -> None: ...

    def process(self, measurement: NDArray[np.floating]) -> dict[str, float]: ...


@dataclass(frozen=True)
class StaticPathPlanner:
    """Minimal planner plugin that returns a prebuilt path."""

    path: NDArray[np.floating]

    def plan(self, *, world: World, standard: SimulationStandard) -> NDArray[np.floating]:
        _ = world, standard
        return self.path.copy()
