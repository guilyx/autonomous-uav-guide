<!-- Erwin Lejeune — 2026-02-23 -->
# Costmap Navigation

## Algorithm

Dynamic costmap with footprint inflation and speed-based dynamic inflation. A* plans on the local costmap at 10 Hz, with moving obstacle avoidance. Dual-view shows global (world frame) and FCU (sensor range) costmaps.

**Reference:** D. Fox, W. Burgard, S. Thrun, "The Dynamic Window Approach to Collision Avoidance," IEEE RA Magazine, 1997.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid resolution | 0.5 m | Costmap cell size |
| Sensor range | 10 m | FCU costmap radius |
| Replan rate | 10 Hz | Path replanning frequency |

## Run

```bash
python -m uav_sim.simulations.environment.costmap_navigation
```
