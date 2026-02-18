# A* 3D Path Planning

Classic graph search on a 3D voxel grid with an admissible Euclidean heuristic. Guaranteed to find the shortest obstacle-free path on the discretised grid. The search is animated node-by-node to show the exploration wavefront.

## Key Equations

$$f(n) = g(n) + h(n), \quad h(n) = \|n - n_{\text{goal}}\|_2$$

## Reference

P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," IEEE Trans. SSC, 1968. [DOI](https://doi.org/10.1109/TSSC.1968.300136)

## Usage

```bash
python -m uav_sim.simulations.path_planning.astar_3d
```

## Result

![astar_3d](astar_3d.gif)
