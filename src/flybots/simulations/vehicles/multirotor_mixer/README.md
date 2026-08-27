# Multirotor Mixer

A quadrotor, a hexacopter and a coaxial octocopter fly the same closed-loop
box under the same cascaded controller. Nothing above the mixer knows how
many rotors are below it: each airframe's allocation matrix is built from
its own rotor positions and spin directions.

## Problem Statement

A hard-coded 4x4 mixing matrix describes exactly one airframe. Given rotor
positions and spin directions, the same matrix can be derived for any
layout — and the quadrotor falls out of it as the four-rotor case.

## Key Equations

Rotor `i` at body position $r_i = (x_i, y_i, z_i)$ in **FLU**, thrusting
along body $+z$ with force $f_i \ge 0$ and spinning with direction
$\sigma_i = \pm 1$:

$$
\begin{bmatrix} T \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}
=
\underbrace{\begin{bmatrix}
1 & \cdots \\ y_i & \cdots \\ -x_i & \cdots \\ -\kappa\sigma_i & \cdots
\end{bmatrix}}_{A \in \mathbb{R}^{4 \times n}}
\begin{bmatrix} f_1 \\ \vdots \\ f_n \end{bmatrix},
\qquad
\kappa = \frac{k_\tau}{k_T}
$$

The mixer is $A^{+}$, the Moore-Penrose pseudo-inverse, which for $n > 4$
picks the minimum-norm thrust vector out of the family that produces the
same wrench.

Note that $z_i$ does not appear. A rotor pushing along $+z$ has no moment
arm about $z$, which is why a coaxial pair adds thrust and yaw authority
but no roll or pitch authority.

The signs come from **FLU**, where positive pitch is nose-down: a rotor at
the front produces $\tau_y = -x_i f_i < 0$, and the nose goes up.

## Reference

M. Achtelik, K.-M. Doth, D. Gurdan, J. Stumpf, "Design of a Multi Rotor MAV
with regard to Efficiency, Dynamics and Redundancy," AIAA Guidance,
Navigation, and Control Conference, 2012.
[DOI](https://doi.org/10.2514/6.2012-4779)

T. A. Johansen, T. I. Fossen, "Control allocation — A survey," Automatica,
49(5):1087-1103, 2013.
[DOI](https://doi.org/10.1016/j.automatica.2013.01.035)

## Usage

```bash
flybots run multirotor_mixer
# or
python -m flybots.simulations.vehicles.multirotor_mixer
```

## Result

![multirotor_mixer](multirotor_mixer.gif)

The lower panel plots each of the hexacopter's six rotor thrusts against
the instantaneous collective. Hover is a flat line. The roll and pitch legs
split the rotors by position. The yaw turn at the end splits them into two
groups of three — by spin direction, not by position, because yaw comes out
of rotor drag and has no lever arm at all.
