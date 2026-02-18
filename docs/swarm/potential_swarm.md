# Potential-Based Swarm Navigation

## Theory

Inter-agent interactions use a Lennard-Jones-like potential:

$$U(r) = \epsilon \left[\left(\frac{d_{\text{des}}}{r}\right)^a - \left(\frac{d_{\text{des}}}{r}\right)^b\right], \quad a > b$$

This creates a repulsive force at close range and attractive force at long range, producing self-organised spacing. A goal potential is added for navigation.

## Reference

W. M. Spears, D. F. Spears, J. C. Hamann, R. Heil, "Distributed, Physics-Based Control of Swarms of Vehicles," Autonomous Robots, 2004. [DOI: 10.1023/B:AURO.0000033971.96584.f2](https://doi.org/10.1023/B:AURO.0000033971.96584.f2)

## Simulation

```bash
uv run python simulations/swarm/potential_swarm/run.py
```

![Potential Swarm](../../simulations/swarm/potential_swarm/potential_swarm.gif)

## API

```python
from uav_sim.swarm.potential_swarm import PotentialSwarm
ctrl = PotentialSwarm(d_des=2.0)
forces = ctrl.compute_forces(positions, goal=goal)
```
