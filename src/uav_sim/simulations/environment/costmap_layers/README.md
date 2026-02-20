# Costmap Layers

Demonstrates layered costmap composition: an occupancy grid from the environment, an inflation layer that grows obstacle boundaries by a configurable radius, and a social layer that penalises areas near moving agents. The layers are fused into a single cost surface for downstream planning.

## Key Equations

$$C_{\text{total}}(x,y) = \max\bigl(C_{\text{occ}}(x,y),\; C_{\text{infl}}(x,y),\; C_{\text{social}}(x,y)\bigr)$$

## Reference

D. V. Lu, D. Hershberger, W. D. Smart, "Layered Costmaps for Context-Sensitive Navigation," IROS 2014. [DOI](https://doi.org/10.1109/IROS.2014.6942636)

## Usage

```bash
python -m uav_sim.simulations.environment.costmap_layers
```

## Result

![costmap_layers](costmap_layers.gif)
