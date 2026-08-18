# Multirotor

Any even rotor count, any arm geometry, one set of equations. The mixing
matrix is **derived** from where the rotors sit and which way they spin,
which is what lets a hexacopter and a coaxial X8 reuse the quadrotor's
dynamics, controllers and estimators unchanged.

> M. Achtelik, K.-M. Doth, D. Gurdan, J. Stumpf, "Design of a Multi Rotor
> MAV with regard to Efficiency, Dynamics and Redundancy", AIAA Guidance,
> Navigation, and Control Conference, 2012.
> [doi:10.2514/6.2012-4779](https://doi.org/10.2514/6.2012-4779)
>
> T. A. Johansen, T. I. Fossen, "Control allocation — A survey",
> Automatica 49(5):1087-1103, 2013.
> [doi:10.1016/j.automatica.2013.01.035](https://doi.org/10.1016/j.automatica.2013.01.035)

Source: [`uav_sim/vehicles/multirotor/`](https://github.com/guilyx/flybots/tree/main/src/uav_sim/vehicles/multirotor)
and [`components/allocation.py`](https://github.com/guilyx/flybots/blob/main/src/uav_sim/vehicles/components/allocation.py)

## Quick start

```python
import numpy as np
from uav_sim.vehicles import VehiclePreset, create_multirotor

craft = create_multirotor(VehiclePreset.HEX_S550)
craft.reset(position=np.array([0.0, 0.0, 5.0]))
craft.spin_up_to_hover()          # motors start stopped; skip the transient

for _ in range(1000):
    craft.step(craft.hover_wrench(), 0.005)

craft.n_rotors            # 6
craft.get_rotor_thrusts() # per-rotor thrust [N]
craft.mixer.rank          # 4 — thrust and all three torques reachable
```

The control input is a body wrench `[T, τx, τy, τz]` whatever the rotor
count, so every controller in `uav_sim.control` works on all of them
without modification.

## Where the mixing matrix comes from

Rotor $i$ sits at body position $r_i = (x_i, y_i, z_i)$ in **FLU** and
pushes along body $+z$ with force $f_i \ge 0$. Its wrench contribution is
the moment of that force about the centre of mass, plus the reaction torque
of its own drag:

$$
F_i = f_i \hat{z}, \qquad
\tau_i = r_i \times F_i - \sigma_i \kappa f_i \hat{z},
\qquad \kappa = \frac{k_\tau}{k_T}
$$

with $\sigma_i = +1$ for a rotor turning counter-clockwise seen from above.
Expanding the cross product gives one column per rotor:

$$
\begin{bmatrix} T \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}
=
\underbrace{\begin{bmatrix}
1      & 1      & \cdots \\
y_1    & y_2    & \cdots \\
-x_1   & -x_2   & \cdots \\
-\kappa\sigma_1 & -\kappa\sigma_2 & \cdots
\end{bmatrix}}_{A \;\in\; \mathbb{R}^{4 \times n}}
\begin{bmatrix} f_1 \\ f_2 \\ \vdots \\ f_n \end{bmatrix}
$$

Three things fall out of this that are worth stating on their own.

**$z_i$ does not appear.** A rotor thrusting along $+z$ has no moment arm
about $z$. Stacking a second rotor under the first therefore adds thrust
and yaw authority and *no roll or pitch authority whatsoever* — an X8 has
the leverage of the quadrotor it is built on.

**Yaw has no lever arm.** Roll and pitch scale with the arm length; yaw
scales with $\kappa$, the torque-to-thrust ratio, which is a property of
the propeller. On the S550 that is 16 mm against a 275 mm arm, so yaw is
seventeen times weaker than roll. This is why yaw slews slowly on every
multirotor you have ever flown, and why the demo ramps its heading command
instead of stepping it.

**The signs are the frame's.** In **FLU**, positive pitch is nose-down (see
[Frames and conventions](/guide/conventions)). A rotor at the front gives
$\tau_y = -x_i f_i < 0$, and the nose goes up. Port these rows from a
Forward-Right-Down textbook and every one of them flips.

### The quadrotor falls out of it

The 4×4 matrices this library hard-coded for its `x` and `+` frames are
reproduced by the derivation to machine precision — that agreement is
pinned by a test, with the historical matrices written out as literals so
the reference cannot drift with the code:

```python
from uav_sim.vehicles.components.mixer import Mixer

Mixer(arm_length=0.175, frame="x").mix_matrix
```

Recovering the layout from the matrix also settled which rotor is which.
The X-frame column order is **rear-left, rear-right, front-right,
front-left**, spinning `CCW, CW, CCW, CW`; the source comment used to claim
it started at the front left.

## Layouts

```python
from uav_sim.vehicles import x_layout, plus_layout, h_layout, coaxial_layout, Rotor
```

| Builder | Shape |
|---|---|
| `x_layout(n, arm)` | Even ring, first rotor half a sector off the tail — nothing on the nose |
| `plus_layout(n, arm)` | Even ring with a rotor on each axis |
| `h_layout(n, length, width)` | Two rails, longer fore-aft than side-to-side |
| `coaxial_layout(base, separation, lower_efficiency)` | Stacks a counter-rotating rotor under each rotor of `base` |

Rotor counts must be **even and at least four**. An odd ring cannot
alternate spin directions all the way round, so it would carry a net yaw
torque at hover — a tricopter solves that with a tilting tail servo, which
is a different actuator model.

Anything the builders do not cover is a list of `Rotor`:

```python
craft = create_multirotor(
    VehiclePreset.CUSTOM,
    mass=2.4,
    rotors=[
        Rotor(position=[0.3, 0.2, 0.0], direction=+1),
        Rotor(position=[0.3, -0.2, 0.0], direction=-1),
        Rotor(position=[-0.3, -0.2, 0.0], direction=+1),
        Rotor(position=[-0.3, 0.2, 0.0], direction=-1),
    ],
)
```

## What the airframe can and cannot do

`ControlAllocation` reports its own limits rather than assuming they are
fine:

```python
craft.mixer.rank                # wrench axes reachable, out of 4
craft.mixer.fully_actuated      # rank == 4
craft.mixer.unreachable_axes    # e.g. ('pitch',) or ('thrust', 'yaw')
craft.mixer.yaw_authority       # Nm of yaw per N of thrust redistributed
craft.mixer.torque_rows_balanced
```

Two failure modes are worth recognising, because both look like working
airframes until you ask them to fly:

**Every rotor spinning the same way.** The yaw row becomes a multiple of
the thrust row. The airframe cannot yaw on command — and cannot hold
altitude without yawing, because every newton it lifts with drags a fixed
reaction torque along with it. `rank` drops to 3 and `unreachable_axes`
reports `('thrust', 'yaw')`.

**Rotors collinear in the `xy` plane.** A coaxial pair on a single lateral
axis has four rotors, four motors and no pitch authority at all. The
pseudo-inverse survives it — a plain matrix inverse would not — and returns
the closest reachable wrench, silently dropping the pitch it cannot make.
`unreachable_axes` is how you find out.

## Saturation: what happens to a negative thrust command

With more than four rotors the allocation is over-determined: a whole
family of thrust vectors produces the same wrench. The mixer resolves that
with the Moore-Penrose pseudo-inverse, which picks the minimum-norm member
and so the least total rotor effort. But minimum-norm is not the same as
*feasible* — a rotor cannot pull, and cannot exceed its maximum thrust, and
the unconstrained solution routinely asks it to.

```python
craft = create_multirotor(VehiclePreset.HEX_S550, saturation="prioritise_torque")
```

| Strategy | What it does | What it costs |
|---|---|---|
| `"clip"` *(default)* | Clamp each rotor into its feasible range | Clamping one rotor perturbs **every** axis of the delivered wrench, attitude included |
| `"prioritise_torque"` | Shift the whole thrust vector by a constant, then clamp | All three torques survive exactly; the collective absorbs the entire error |

The shift works because on a balanced layout the roll, pitch and yaw rows
each sum to zero, so adding the same amount to every rotor changes the
total thrust and nothing else. This is the "air mode" of the open-source
flight stacks, and it is the right trade: an aircraft that loses a little
altitude authority recovers, and one that loses roll authority does not.
The default stays `"clip"` because that is what the library has always
done and a quadrotor's behaviour had to be preserved exactly.

Either way the request and the result are recorded:

```python
forces = craft.mixer.wrench_to_forces(wrench)
craft.mixer.last_saturation.error             # achieved - requested, per axis
craft.mixer.last_saturation.collective_shift  # newtons of shift applied
```

## Presets

```python
craft = create_multirotor(VehiclePreset.HEX_S550)     # 1.8 kg flat hex
craft = create_multirotor(VehiclePreset.OCTO_X8)      # 4.5 kg coaxial octo
craft = create_multirotor(VehiclePreset.RACING_250)   # returns a Quadrotor
```

| Preset | Rotors | Mass | Arm | T/W | Layout |
|---|---|---|---|---|---|
| `HEX_S550` | 6 | 1.8 kg | 275 mm | 3.0 | Flat hexa X |
| `OCTO_X8` | 8 | 4.5 kg | 350 mm | 2.6 | Coaxial, 4 arms |

Inertias are built from a lumped model — point masses at the rotor hubs
plus a central disc — rather than guessed, so `I_zz ≈ 2 I_xx` comes out of
the layout instead of being asserted.

The X8's lower rotors work in the wake of the upper ones and return about
85% of their thrust (Leishman, *Principles of Helicopter Aerodynamics*,
Sec. 2.14). That is modelled as a per-rotor `thrust_scale` on both the
thrust and the torque curve — a first-order lump, not a wake model — and it
is why `thrust_to_weight` does not credit the airframe with eight
clean-air rotors, and why the lower rotor of each pair runs faster than the
upper one in hover.

## Quadrotors

`Quadrotor` is a preset over `Multirotor`, not a separate model:

```python
from uav_sim.vehicles import Quadrotor, create_quadrotor, VehiclePreset

isinstance(create_quadrotor(VehiclePreset.CRAZYFLIE), Multirotor)   # True
```

Its parameters still describe the geometry as an arm length plus a frame
letter, because for four rotors that is a complete description. Everything
else on the page applies to it unchanged — see
[Quadrotor](/vehicles/quadrotor) for the presets and the control stack.

## See also

- [Multirotor mixer simulation](/simulations/vehicles/multirotor-mixer)
- [Quadrotor](/vehicles/quadrotor)
- [Frames and conventions](/guide/conventions)
