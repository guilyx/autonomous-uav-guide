# Reynolds Flocking

## Theory

Three simple rules produce emergent flock behaviour:

1. **Separation**: steer away from nearby neighbours within radius $r_{\text{sep}}$.
2. **Alignment**: match the average velocity of neighbours within $r_{\text{percept}}$.
3. **Cohesion**: steer towards the average position of neighbours within $r_{\text{percept}}$.

The total force is a weighted sum: $F = w_s F_{\text{sep}} + w_a F_{\text{ali}} + w_c F_{\text{coh}}$.

## Reference

C. W. Reynolds, "Flocks, Herds and Schools: A Distributed Behavioral Model," SIGGRAPH '87, 1987. [DOI: 10.1145/37402.37406](https://doi.org/10.1145/37402.37406)

## Simulation

```bash
uv run python simulations/reynolds_flocking/run.py
```

![Reynolds Flocking](../../simulations/reynolds_flocking/reynolds_flocking.gif)

## API

```python
from quadrotor_sim.swarm.reynolds_flocking import ReynoldsFlocking
flock = ReynoldsFlocking()
forces = flock.compute_forces(positions, velocities)  # (N,3) each
```
