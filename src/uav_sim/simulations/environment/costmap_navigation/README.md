# Dynamic Costmap Navigation

Reactive obstacle avoidance using a local costmap that is rebuilt every
planning cycle.  The costmap includes:

- **Footprint inflation** — obstacles expanded by the vehicle bounding radius.
- **Speed-based dynamic inflation** — safety margin scales with drone speed
  to account for braking distance.
- **Dynamic obstacles** — three moving agents that the drone must avoid.

## Key Equations

$$f(n) = g(n) + h(n), \quad h(n) = \| n - n_{\text{goal}} \|$$

Inflation radius:

$$r_{\text{infl}} = r_{\text{footprint}} + r_{\text{pad}} + k_v \cdot v$$

## Reference

D. Fox, W. Burgard, S. Thrun, "The Dynamic Window Approach to Collision
Avoidance," IEEE RA Magazine, 1997.

## Usage

```bash
python -m uav_sim.simulations.environment.costmap_navigation
```

## Result

![costmap_navigation](costmap_navigation.gif)
