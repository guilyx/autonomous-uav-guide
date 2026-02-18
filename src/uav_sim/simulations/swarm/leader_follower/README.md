# Leader-Follower Formation

A designated leader follows a reference trajectory while followers maintain specified offsets using attractive/repulsive potential fields. Collision avoidance is built into the inter-agent potential.

## Key Equations

$$u_i = k_{\text{track}}(p_{\text{leader}} + d_i - p_i) + \sum_{j \\neq i} f_{\text{rep}}(p_i, p_j)$$

## Reference

J. P. Desai, J. P. Ostrowski, V. Kumar, "Modeling and Control of Formations of Nonholonomic Mobile Robots," IEEE TRA, 2001. [DOI](https://doi.org/10.1109/70.976023)

## Usage

```bash
python -m uav_sim.simulations.swarm.leader_follower
```

## Result

![leader_follower](leader_follower.gif)
