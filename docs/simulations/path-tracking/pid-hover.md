<!-- Erwin Lejeune — 2026-02-24 -->
# PID Hover

## Problem Statement

PID hover control provides a robust baseline for multirotor stabilization and waypoint holding.
It decomposes position and attitude regulation into cascaded loops with straightforward gain interpretation.

## Model and Formulation

Position error:

$$
e_p = p_{ref} - p,\quad e_v = v_{ref} - v
$$

Outer-loop acceleration command:

$$
a_c = K_p e_p + K_d e_v + K_i \int e_p\,dt
$$

The command is mapped to attitude and thrust for the inner loop.

## Algorithm Procedure

1. Compute position and velocity errors from state estimate.
2. Generate desired acceleration from PID law.
3. Convert acceleration command to desired roll, pitch, and thrust.
4. Stabilize attitude with inner-loop PID and actuator mixing.

## Tuning Guidance

- Tune `K_d` before increasing `K_p` to avoid oscillation.
- Integrator anti-windup is required near thrust saturation.
- Set vertical-axis gains independently from lateral gains.

## Failure Modes and Diagnostics

- High `K_i` can cause slow oscillation and overshoot after disturbances.
- Actuator saturation can destabilize attitude during large position errors.
- Check phase margins by injecting small-step position commands.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_tracking.pid_hover
```

## Evidence

![PID Hover](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif)

## References

- [Carrillo et al., Quad Rotorcraft Control](https://link.springer.com/book/10.1007/978-1-4471-4399-4)
- [Bouabdallah, Design and Control of Quadrotors (EPFL Thesis)](https://infoscience.epfl.ch/record/95939)

## Related Algorithms

- [LQR Hover](/simulations/path-tracking/lqr-hover)
- [Feedback Linearisation](/simulations/trajectory-tracking/feedback-linearisation)
