# Perception Algorithms

## Overview

Perception modules consume sensor measurements to build maps and detect obstacles.

| Module | Description |
| --- | --- |
| `OccupancyMapper` | Incremental log-odds occupancy mapping from lidar |
| `RangeObstacleDetector` | Cluster-based obstacle detection from range data |
| `point_cloud` | Ranges→cloud, voxel downsample, ground removal |

## Reference

S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics," MIT Press, 2005, Chapter 6.

## Simulation

```bash
uv run python simulations/perception/lidar_mapping/run.py
```

![Lidar Mapping](../../simulations/perception/lidar_mapping/lidar_mapping.gif)

## API

```python
from uav_sim.perception import OccupancyMapper, RangeObstacleDetector
mapper = OccupancyMapper(grid)
mapper.update(position, ranges, angles, max_range)

detector = RangeObstacleDetector(threshold=10.0)
centroids = detector.detect(ranges, angles)
```
