<!-- Erwin Lejeune — 2026-02-23 -->
# PID Hover

## Algorithm

Cascaded PID controller with outer position loop and inner attitude loop. The state machine handles takeoff, fly-to-target, and hover phases.

**Reference:** L. R. G. Carrillo et al., "Quad Rotorcraft Control," Springer, 2013.

## Run

```bash
python -m uav_sim.simulations.path_tracking.pid_hover
```
