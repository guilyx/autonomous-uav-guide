# Environment System

## Overview

The environment system provides the simulation world: obstacles, buildings, and dynamic agents. It supports both indoor (bounded room) and outdoor (open area) scenarios.

## Components

| Module | Description |
| --- | --- |
| `World` | Container for obstacles, bounds, and dynamic agents |
| `SphereObstacle` | Spherical obstacle with collision/distance queries |
| `BoxObstacle` | Axis-aligned box obstacle |
| `CylinderObstacle` | Vertical cylinder obstacle |
| `DynamicAgent` | Moving entity (other drones, vehicles) |
| `add_city_grid()` | Procedural urban building generator |

## API

```python
from uav_sim.environment import World, SphereObstacle, WorldType
world = World(bounds_min=np.zeros(3), bounds_max=np.full(3, 50.0), world_type=WorldType.OUTDOOR)
world.add_obstacle(SphereObstacle(centre=np.array([10, 10, 5]), radius=2.0))
assert world.is_free(np.array([0, 0, 0]))
```
