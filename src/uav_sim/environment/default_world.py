# Erwin Lejeune - 2026-02-15
"""Environment factories and presets.

Provides ready-made environments (city, indoor, open field) with
matched drone scale suggestions. Every simulation should use one of
these factories to ensure consistent, comparable GIFs.

Usage::

    world, buildings = default_world()
    world, buildings = create_environment(EnvironmentPreset.INDOOR)
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from uav_sim.environment.buildings import add_urban_buildings
from uav_sim.environment.obstacles import BoxObstacle
from uav_sim.environment.world import World


class EnvironmentPreset(Enum):
    """Catalogue of environment presets."""

    CITY = "city"
    INDOOR = "indoor"
    OPEN_FIELD = "open_field"


def default_world(
    world_size: float = 30.0,
    n_buildings: int = 6,
    seed: int = 42,
) -> tuple[World, list[BoxObstacle]]:
    """Create the standard 30x30x30 urban simulation environment.

    Returns
    -------
    world : the World object.
    buildings : list of BoxObstacle instances for visualization.
    """
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, world_size),
    )
    buildings = add_urban_buildings(
        world,
        world_size=world_size,
        n_buildings=n_buildings,
        seed=seed,
    )
    return world, buildings


def city_world(
    world_size: float = 50.0,
    n_buildings: int = 12,
    seed: int = 42,
) -> tuple[World, list[BoxObstacle]]:
    """Dense urban environment (50x50x50 m) — for DJI-class drones."""
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, world_size),
    )
    buildings = add_urban_buildings(
        world,
        world_size=world_size,
        n_buildings=n_buildings,
        seed=seed,
    )
    return world, buildings


def indoor_world(
    room_size: float = 10.0,
    seed: int = 7,
) -> tuple[World, list[BoxObstacle]]:
    """Small indoor room (10x10x3 m) — for Crazyflie-class drones.

    Creates walls as thin box obstacles and a few interior pillars.
    """
    rng = np.random.RandomState(seed)
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.array([room_size, room_size, 3.0]),
    )
    obstacles: list[BoxObstacle] = []
    n_pillars = rng.randint(2, 5)
    for _ in range(n_pillars):
        cx = rng.uniform(2.0, room_size - 2.0)
        cy = rng.uniform(2.0, room_size - 2.0)
        sz = rng.uniform(0.3, 0.8)
        pillar = BoxObstacle(
            min_corner=np.array([cx - sz, cy - sz, 0.0]),
            max_corner=np.array([cx + sz, cy + sz, 3.0]),
        )
        world.add_obstacle(pillar)
        obstacles.append(pillar)
    return world, obstacles


def open_field(
    field_size: float = 60.0,
) -> tuple[World, list[BoxObstacle]]:
    """Large open field (60x60x30 m) — no obstacles, for testing."""
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.array([field_size, field_size, 30.0]),
    )
    return world, []


def create_environment(
    preset: EnvironmentPreset = EnvironmentPreset.CITY,
    **kwargs: float,
) -> tuple[World, list[BoxObstacle]]:
    """Create an environment from a named preset.

    Parameters
    ----------
    preset : which environment to build.
    **kwargs : forwarded to the underlying factory.

    Returns
    -------
    (world, buildings) tuple.
    """
    factories = {
        EnvironmentPreset.CITY: city_world,
        EnvironmentPreset.INDOOR: indoor_world,
        EnvironmentPreset.OPEN_FIELD: open_field,
    }
    return factories[preset](**kwargs)
