# Boustrophedon Coverage Path Planning

Generates a lawnmower (boustrophedon) sweep pattern that ensures full area coverage at a given altitude. Swath width is computed from camera FOV and altitude. After planning, the drone follows the coverage path using Pure Pursuit.

## Key Equations

$$w_{\text{swath}} = 2\,h\,\tan\!\left(\frac{\text{FOV}}{2}\right)(1 - \text{overlap})$$

## Reference

H. Choset, "Coverage of Known Spaces: The Boustrophedon Cellular Decomposition," Autonomous Robots, 2000. [DOI](https://doi.org/10.1023/A:1008958800904)

## Usage

```bash
python -m uav_sim.simulations.path_planning.coverage_planning
```

## Result

![coverage_planning](coverage_planning.gif)
