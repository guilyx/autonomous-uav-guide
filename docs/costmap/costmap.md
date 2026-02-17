# Costmap System

## Overview

Layered costmaps for planning and navigation. The base occupancy grid is rasterised from the World, then additional layers (inflation, social) compose on top.

## Layers

| Layer | Description |
| --- | --- |
| `OccupancyGrid` | 2D/3D binary/probabilistic occupancy |
| `InflationLayer` | Exponential cost decay around obstacles |
| `SocialLayer` | Velocity-dependent cost for dynamic agents |
| `LayeredCostmap` | Compositor: queries all layers |

## Reference

S. Thrun, "Learning Occupancy Grid Maps with Forward Sensor Models," Autonomous Robots, 2003.

## Simulation

```bash
uv run python simulations/environment/costmap_navigation/run.py
```

![Costmap Navigation](../../simulations/environment/costmap_navigation/costmap_navigation.gif)

## API

```python
from uav_sim.costmap import OccupancyGrid, InflationLayer, LayeredCostmap
grid = OccupancyGrid(resolution=0.5)
grid.from_world(world)
costmap = LayeredCostmap(grid, inflation=InflationLayer(inflation_radius=2.0))
costmap.update()
cost = costmap.cost_at(point)
```
