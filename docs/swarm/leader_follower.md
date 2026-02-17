# Leader-Follower Formation

## Theory

A designated leader tracks an arbitrary trajectory; followers maintain desired offsets from the leader via PD control:

$$F_i = k_p(p_L + \delta_i - p_i) + k_d(\dot{p}_L - \dot{p}_i)$$

Simple, robust, and easy to implement. The single point of failure is the leader.

## Reference

J. P. Desai, J. P. Ostrowski, V. Kumar, "Modeling and Control of Formations of Nonholonomic Mobile Robots," IEEE T-RA, 2001. [DOI: 10.1109/70.976023](https://doi.org/10.1109/70.976023)

## Simulation

```bash
uv run python simulations/leader_follower/run.py
```

![Leader-Follower](../../simulations/leader_follower/leader_follower.gif)

## API

```python
from quadrotor_sim.swarm.leader_follower import LeaderFollower
ctrl = LeaderFollower(offsets=offsets, kp=4.0, kd=2.0)
forces = ctrl.compute_forces(leader_pos, leader_vel, follower_pos, follower_vel)
```
