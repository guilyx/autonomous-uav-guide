# Voronoi Coverage Control

Agents iteratively move toward the centroid of their Voronoi cell, distributing themselves optimally to cover a region. Converges to a centroidal Voronoi tessellation that minimises a coverage cost functional.

## Key Equations

$$\dot p_i = k(C_{V_i} - p_i), \quad C_{V_i} = \frac{\int_{V_i} q\,\phi(q)\,dq}{\int_{V_i} \phi(q)\,dq}$$

## Reference

J. Cortes, S. Martinez, T. Karatas, F. Bullo, "Coverage Control for Mobile Sensing Networks," IEEE TRA, 2004. [DOI](https://doi.org/10.1109/TRA.2004.838006)

## Usage

```bash
python -m uav_sim.simulations.swarm.voronoi_coverage
```

## Result

![voronoi_coverage](voronoi_coverage.gif)
