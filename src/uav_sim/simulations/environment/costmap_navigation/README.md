# Costmap Navigation

End-to-end demonstration of planning a collision-free path through a layered costmap using A* on the inflated grid, then tracking that path with a PID controller.

## Key Equations

$$f(n) = g(n) + h(n), \quad h(n) = \| n - n_{\text{goal}} \|$$

## Reference

P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," IEEE Trans. SSC, 1968. [DOI](https://doi.org/10.1109/TSSC.1968.300136)

## Usage

```bash
python -m uav_sim.simulations.environment.costmap_navigation
```

## Result

![costmap_navigation](costmap_navigation.gif)
