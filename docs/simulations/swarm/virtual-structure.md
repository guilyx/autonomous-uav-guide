<!-- Erwin Lejeune — 2026-02-23 -->
# Virtual Structure Formation

## Algorithm

Agents track assigned positions on a moving virtual rigid body. The structure follows a circular path and agents use PD control to maintain their body-frame offsets.

**Reference:** M. A. Lewis, K.-H. Tan, "High Precision Formation Control of Mobile Robots Using Virtual Structures," Autonomous Robots, 1997.

## Run

```bash
python -m uav_sim.simulations.swarm.virtual_structure
```
