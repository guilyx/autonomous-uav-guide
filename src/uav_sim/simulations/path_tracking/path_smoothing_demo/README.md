# Path Smoothing

Compares raw A*-planned paths with two smoothing techniques: Ramer-Douglas-Peucker simplification (reduces waypoints) and cubic-spline smoothing (creates continuous curvature). The drone follows the smoothed path with PID control.

## Key Equations

$$\text{RDP: remove } p_i \text{ if } d(p_i, \overline{p_0 p_n}) < \epsilon$$

## Reference

D. Douglas, T. Peucker, "Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line," Cartographica, 1973. [DOI](https://doi.org/10.3138/FM57-6770-U75U-7727)

## Usage

```bash
python -m uav_sim.simulations.path_tracking.path_smoothing_demo
```

## Result

![path_smoothing_demo](path_smoothing_demo.gif)
