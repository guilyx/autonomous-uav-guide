# Cascaded PID Controller

## Theory

A two-loop cascade: an outer position loop generates attitude references, and an inner attitude loop stabilises orientation. Each axis uses a standard PID law:

$$u = K_p e + K_i \int e\,dt + K_d \dot{e}$$

The outer loop converts position error into desired roll/pitch, which the inner loop tracks.

## Reference

L. R. G. Carrillo, A. E. D. López, R. Lozano, C. Pégard, "Quad Rotorcraft Control: Vision-Based Hovering and Navigation," Springer, 2013, Chapter 4.

## Simulation

```bash
uv run python simulations/path_tracking/pid_hover/run.py
```

![PID Hover](../../simulations/path_tracking/pid_hover/pid_hover.gif)

## API

```python
from uav_sim.path_tracking.pid_controller import CascadedPIDController
ctrl = CascadedPIDController()
wrench = ctrl.compute(state, target_position, dt=0.002)
```
