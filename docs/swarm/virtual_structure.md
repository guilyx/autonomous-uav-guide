# Virtual Structure Formation

## Theory

A virtual rigid body moves along a desired trajectory. Each agent tracks a fixed offset from this body's pose using PD control:

$$F_i = k_p(p_{d,i} - p_i) - k_d v_i$$

where $p_{d,i} = p_{\text{body}} + R(\psi)\delta_i$ is the desired position for agent $i$.

## Reference

M. A. Lewis, K.-H. Tan, "High Precision Formation Control of Mobile Robots Using Virtual Structures," Autonomous Robots, 1997. [DOI: 10.1023/A:1008814708459](https://doi.org/10.1023/A:1008814708459)

## Simulation

```bash
uv run python simulations/swarm/virtual_structure/run.py
```

![Virtual Structure](../../simulations/swarm/virtual_structure/virtual_structure.gif)

## API

```python
from uav_sim.swarm.virtual_structure import VirtualStructure
ctrl = VirtualStructure(body_offsets=offsets, kp=3.0, kd=2.0)
forces = ctrl.compute_forces(positions, velocities, body_pos, body_yaw)
```
