<!-- Erwin Lejeune — 2026-02-24 -->
# Quadrotor Dynamics

## Problem Statement

Quadrotor dynamics modeling provides the foundational equations used by estimators, planners, and controllers.

## Model and Formulation

Translational dynamics:

$$
m\ddot{p}=mg + R(\phi,\theta,\psi)\begin{bmatrix}0\\0\\T\end{bmatrix}
$$

Rotational dynamics:

$$
I\dot{\omega} + \omega \times I\omega = \tau
$$

with thrust `T` and body torque vector `\tau`.

## Practical Notes

- Model fidelity depends on aerodynamic drag and propeller assumptions.
- Hover linearization is valid only near small angles and low rates.
- Actuator limits and delays must be represented for realistic control studies.

## Evidence

![Quadrotor Dynamics](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif)

## References

- [Bouabdallah, Design and Control of Quadrotors (EPFL Thesis)](https://infoscience.epfl.ch/record/95939)
- [Beard and McLain, Small Unmanned Aircraft](https://press.princeton.edu/books/hardcover/9780691149219/small-unmanned-aircraft)

## Related Algorithms

- [PID Hover](/simulations/path-tracking/pid-hover)
- [LQR Hover](/simulations/path-tracking/lqr-hover)
