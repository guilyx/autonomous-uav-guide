<!-- Erwin Lejeune — 2026-02-23 -->
# Reynolds Flocking

## Algorithm

Emergent collective motion from three simple rules applied to each agent: separation (avoid crowding), alignment (steer toward average heading), cohesion (steer toward average position).

**Reference:** C. W. Reynolds, "Flocks, Herds, and Schools: A Distributed Behavioral Model," SIGGRAPH, 1987.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Agents | 8 | Flock size |
| r_percept | 40 m | Perception radius |
| r_sep | 8 m | Separation radius |

## Run

```bash
python -m uav_sim.simulations.swarm.reynolds_flocking
```
