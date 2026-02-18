# Probabilistic Roadmap (PRM) 3D

Samples random collision-free configurations in 3D space, connects nearby samples with edges verified for collision, and queries the resulting roadmap graph with Dijkstra's algorithm. Good for multi-query planning in static environments.

## Key Equations

$$\text{Connect } (q_i, q_j) \iff \|q_i - q_j\| < r \;\wedge\; \text{collision\_free}(q_i, q_j)$$

## Reference

L. E. Kavraki et al., "Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces," IEEE TRA, 1996. [DOI](https://doi.org/10.1109/70.508439)

## Usage

```bash
python -m uav_sim.simulations.path_planning.prm_3d
```

## Result

![prm_3d](prm_3d.gif)
