# Erwin Lejeune - 2026-02-17
"""World container for obstacles, boundaries, and dynamic agents.

Supports both indoor (bounded room) and outdoor (open area with buildings)
environments. All path planners and costmaps query the World for collision
checks and distance queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.obstacles import Obstacle


class WorldType(Enum):
    INDOOR = auto()
    OUTDOOR = auto()


@dataclass
class World:
    """Simulation world with static and dynamic obstacles.

    Parameters
    ----------
    bounds_min, bounds_max:
        Axis-aligned bounding box of the navigable space.
    world_type:
        Indoor or outdoor semantics.
    obstacles:
        Static obstacles populating the world.
    """

    bounds_min: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))
    bounds_max: NDArray[np.floating] = field(default_factory=lambda: np.full(3, 50.0))
    world_type: WorldType = WorldType.OUTDOOR
    obstacles: list[Obstacle] = field(default_factory=list)
    dynamic_agents: list[DynamicAgent] = field(default_factory=list)

    # ── queries ───────────────────────────────────────────────────────────

    def in_bounds(self, point: NDArray[np.floating]) -> bool:
        return bool(np.all(point >= self.bounds_min) and np.all(point <= self.bounds_max))

    def is_free(self, point: NDArray[np.floating]) -> bool:
        """Return True if *point* is inside bounds and outside all obstacles."""
        if not self.in_bounds(point):
            return False
        return all(not o.contains(point) for o in self.obstacles)

    def nearest_obstacle_distance(self, point: NDArray[np.floating]) -> float:
        if not self.obstacles:
            return float("inf")
        return min(o.distance(point) for o in self.obstacles)

    def add_obstacle(self, obs: Obstacle) -> None:
        self.obstacles.append(obs)

    # ── dynamic agents ────────────────────────────────────────────────────

    def add_agent(self, agent: DynamicAgent) -> None:
        self.dynamic_agents.append(agent)

    def step_agents(self, dt: float) -> None:
        for a in self.dynamic_agents:
            a.step(dt)


@dataclass
class DynamicAgent:
    """A moving object in the world (another drone, pedestrian, vehicle)."""

    position: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))
    velocity: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.5

    def step(self, dt: float) -> None:
        self.position = self.position + self.velocity * dt
