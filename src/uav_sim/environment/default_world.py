# Erwin Lejeune - 2026-02-15
"""Canonical 30x30x30 world factory for consistent simulations.

Every simulation should call :func:`default_world` to get the same building
layout, deterministic seed, and 30 m boundaries — ensuring that GIFs look
consistent and algorithms are directly comparable.
"""

from __future__ import annotations

import numpy as np

from uav_sim.environment.buildings import add_urban_buildings
from uav_sim.environment.obstacles import BoxObstacle
from uav_sim.environment.world import World


def default_world(
    world_size: float = 30.0,
    n_buildings: int = 6,
    seed: int = 42,
) -> tuple[World, list[BoxObstacle]]:
    """Create the standard simulation environment.

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
