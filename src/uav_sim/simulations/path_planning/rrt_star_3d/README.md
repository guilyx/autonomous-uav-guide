# RRT* 3D

Asymptotically optimal rapidly-exploring random tree. Incrementally builds a tree of collision-free paths, rewiring to reduce cost. Converges to the optimal path given enough samples.

## Key Equations

$$r_n = \gamma\left(\frac{\log n}{n}\right)^{1/d}, \quad \text{cost}(x) = \min_{x' \in \text{Near}} \text{cost}(x') + \|x' - x\|$$

## Reference

S. Karaman, E. Frazzoli, "Sampling-Based Algorithms for Optimal Motion Planning," IJRR, 2011. [DOI](https://doi.org/10.1177/0278364911406761)

## Usage

```bash
python -m uav_sim.simulations.path_planning.rrt_star_3d
```

## Result

![rrt_star_3d](rrt_star_3d.gif)
