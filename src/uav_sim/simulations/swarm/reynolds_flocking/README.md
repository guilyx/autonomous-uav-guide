# Reynolds Flocking

Implements Craig Reynolds' three flocking rules: separation (avoid crowding neighbours), alignment (steer toward average heading), and cohesion (steer toward average position). The emergent flock behaviour resembles bird or fish schools.

## Key Equations

$$a_i = w_s\,a_{\text{sep}} + w_a\,a_{\text{align}} + w_c\,a_{\text{coh}}$$

## Reference

C. W. Reynolds, "Flocks, Herds and Schools: A Distributed Behavioral Model," SIGGRAPH 1987. [DOI](https://doi.org/10.1145/37401.37406)

## Usage

```bash
python -m uav_sim.simulations.swarm.reynolds_flocking
```

## Result

![reynolds_flocking](reynolds_flocking.gif)
