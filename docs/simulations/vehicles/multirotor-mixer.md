<!-- Erwin Lejeune — 2026-08-18 -->
# Multirotor Mixer

## Problem Statement

A hard-coded 4×4 mixing matrix describes exactly one airframe. Given rotor
positions and spin directions, the same matrix can be derived for any
layout — and the quadrotor falls out of it as the four-rotor case.

This run flies a quadrotor, a hexacopter and a coaxial octocopter through
the same closed-loop box under the same cascaded controller. Nothing above
the mixer knows how many rotors are below it.

## Model and Formulation

Rotor $i$ at body position $r_i = (x_i, y_i, z_i)$ in FLU, thrusting along
body $+z$ with force $f_i \ge 0$ and spin direction $\sigma_i = \pm 1$:

$$
\begin{bmatrix} T \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}
= A f, \qquad
A_{:,i} = \begin{bmatrix} 1 \\ y_i \\ -x_i \\ -\kappa \sigma_i \end{bmatrix},
\qquad \kappa = \frac{k_\tau}{k_T}
$$

The mixer is $A^{+}$, the Moore-Penrose pseudo-inverse. For four rotors $A$
is square and $A^{+}$ is its inverse; for more, $A^{+}$ picks the
minimum-norm thrust vector out of the family that produces the same wrench.

The rate loop gains are scaled by each airframe's inertia so all three fly
at the same closed-loop bandwidth — constant angular acceleration per unit
rate error, which reproduces the library's default quadrotor gains exactly.

## Practical Notes

- $z_i$ never enters $A$, so a coaxial pair adds thrust and yaw authority
  and no roll or pitch authority. The X8 has the leverage of the quadrotor
  it is built on.
- Yaw has no lever arm: it scales with $\kappa$ (16 mm on the S550) against
  a 275 mm roll arm. The heading command is ramped rather than stepped,
  because a 90° step saturates every rotor before the aircraft has turned
  ten degrees.
- The signs come from FLU, where positive pitch is nose-down. A rotor at
  the front produces $\tau_y < 0$ and the nose goes up.

## Evidence

![Multirotor Mixer](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/vehicles/multirotor_mixer/multirotor_mixer.gif)

The lower panel plots each of the hexacopter's six rotor thrusts against
the instantaneous collective. Hover is a flat line. The translation legs
split the rotors by position. The yaw turn splits them into two groups of
three by **spin direction**, which is where yaw authority comes from.

All three airframes hold their waypoints to about 0.34 m and reach the
commanded 90° heading, with roughly 2× thrust headroom throughout.

## References

- [Achtelik et al., Design of a Multi Rotor MAV with regard to Efficiency, Dynamics and Redundancy (AIAA GNC 2012)](https://doi.org/10.2514/6.2012-4779)
- [Johansen and Fossen, Control allocation — A survey (Automatica 2013)](https://doi.org/10.1016/j.automatica.2013.01.035)
- [Mahony, Kumar and Corke, Multirotor Aerial Vehicles (IEEE RAM 2012)](https://doi.org/10.1109/MRA.2012.2206474)

## Related Algorithms

- [Quadrotor Dynamics](/simulations/vehicles/quadrotor-dynamics)
- [PID Hover](/simulations/path-tracking/pid-hover)
- [Multirotor flight model](/vehicles/multirotor)
