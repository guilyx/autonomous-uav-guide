# VTOL tilt-rotor

A wing plus rotors whose thrust axis rotates from vertical (hover) to
horizontal (cruise). The interesting part is not either end — it is the
**transition**, where lift authority migrates from the rotors to the wing
while the aircraft has to keep flying.

Source: [`uav_sim/vehicles/vtol/`](https://github.com/guilyx/flybots/tree/main/src/uav_sim/vehicles/vtol)

## Quick start

```python
import numpy as np
from uav_sim.vehicles.vtol import Tiltrotor
from uav_sim.control.vtol_controller import VTOLController, VTOLCommand

vtol = Tiltrotor()
state = np.zeros(12)
state[2] = 25.0
vtol.reset(state=state)

pilot = VTOLController(vtol.vtol_params)
command = VTOLCommand(altitude=25.0, cruise=True, cruise_airspeed=24.0)

for _ in range(9000):
    vtol.step(pilot.compute(vtol.state, vtol.tilt, command, 0.01), 0.01)

print(pilot.mode)              # VTOLMode.CRUISE
print(vtol.lift_fraction)      # ~0.99 — the wing carries the aircraft
print(vtol.state[2])           # ~25.0 — altitude held throughout
```

## State and control

```text
state   = [x, y, z, φ, θ, ψ, vx, vy, vz, p, q, r]
control = [thrust, τx, τy, τz, tilt]
```

Velocity is world **ENU**, matching the quadrotor — a VTOL is closer to a
multirotor in how it is commanded. `thrust` is total rotor thrust along the
tilted axis, the torques are body-frame control moments from differential
rotor thrust and surfaces, and `tilt` is the **commanded** rotor angle.

Tilt is slew-rate limited, so a step command cannot teleport the rotors:

```python
vtol.tilt        # actual tilt after rate limiting
vtol.vtol_params.tilt_rate_limit    # 15 deg/s by default
```

`HOVER_TILT` is 0 (thrust up along body `+z`); `CRUISE_TILT` is π/2 (thrust
forward along body `+x`).

## The wing

The tilt-rotor reuses the fixed-wing
[airframe model](/vehicles/fixed-wing) via `airframe_wrench()`, with a
VTOL-appropriate coefficient set: a thicker, more lightly loaded wing that
stalls earlier.

That reuse is the point. The wing does not care that the rotors exist.

```python
vtol.airspeed
vtol.alpha              # from the body-relative airflow
vtol.beta
vtol.wing_lift          # [N] from the last dynamics evaluation
vtol.lift_fraction      # share of weight carried by the wing, 0 -> 1
vtol.vtol_params.stall_airspeed
```

`lift_fraction` is the single most useful number for watching a
transition: it runs from ~0 in hover to ~1 once wing-borne.

## Three things this model gets right

These are worth stating explicitly, because they are the three things the
earlier version of this model got wrong, and each is easy to get wrong
again.

### Angle of attack is body-relative

Incidence is measured between the wing and the airflow **as seen from the
body**, so pitch attitude changes it:

```python
# Level flight at 20 m/s, held 10° nose-up
vtol.alpha    # ≈ +10°, not 0
```

Computing α from the world-frame flight path instead makes pitch attitude
invisible to the wing — the aircraft can point anywhere and the wing never
notices.

### Wing lift comes from airspeed, not from rotor tilt

```python
# Same 20 m/s, three different rotor tilts
tilt =  0°  ->  lift = 75.95 N
tilt = 45°  ->  lift = 75.95 N
tilt = 90°  ->  lift = 75.95 N
```

The wing does not know where the rotors point. Gating lift on tilt is a
tempting shortcut — it stops the wing "cheating" in hover — but the real
reason a wing produces no lift in hover is that there is no airflow, and
$\bar q = \tfrac12\rho V_a^2$ already handles that:

| Airspeed | Wing lift | Share of weight |
|---|---|---|
| 0 m/s | 0.00 N | 0% |
| 5 m/s | 4.75 N | 10% |
| 10 m/s | 18.99 N | 39% |
| 15 m/s | 42.72 N | 87% |
| 20 m/s | 75.95 N | 155% |

That quadratic build-up *is* the transition mechanism.

### Lift tilts with bank, so the aircraft turns

Lift acts perpendicular to the relative wind in the body's plane of
symmetry. Bank the aircraft and the lift vector banks with it, producing
the horizontal component that turns it:

| Bank | Lateral acceleration |
|---|---|
| 0° | 0.00 m/s² |
| 20° | −4.50 m/s² |
| 40° | −7.28 m/s² |

If lift is instead pinned to world-vertical, banking does nothing and
coordinated turns are impossible.

## The transition controller

`VTOLController` runs an explicit mode machine:

```text
HOVER ──airspeed builds──▶ TRANSITION ──wing-borne──▶ CRUISE
  ▲                                                     │
  └──────────── BACK_TRANSITION ◀───────decelerate──────┘
```

### Control allocation flips through the transition

Rotor-borne and wing-borne flight allocate the two actuators to the two
objectives in opposite ways:

| | Altitude held by | Airspeed set by |
|---|---|---|
| Hover | rotor thrust | pitch attitude |
| Cruise | pitch attitude | rotor thrust |

The controller blends between them on a weight that depends on **both**
airspeed and actual rotor tilt, taking whichever is lower. Airspeed alone
is not enough: the rotors slew at a finite rate, so there is a window where
the aircraft is fast enough for the wing but the rotors still point mostly
up. Handing over on airspeed alone in that window leaves nobody holding the
aircraft up.

### Cruise needs a trim feed-forward

A pure error-driven pitch loop commands zero pitch at zero altitude error —
which is exactly the attitude at which a wing produces too little lift to
stay up. The controller feeds forward the incidence the wing actually needs:

$$
C_{L,required} = \frac{2mg}{\rho V_a^2 S},\qquad
\alpha_{trim} = \frac{C_{L,required} - C_{L_0}}{C_{L_\alpha}}
$$

capped below stall, because past the stall boundary more incidence means
*less* lift.

### The attitude loop cancels the wing's own moment

At cruise speed the wing's static stability produces a pitching moment an
order of magnitude larger than a hover-sized PD controller can generate.
The controller already computes the wing wrench for the altitude law, so it
feeds it forward and cancels it, leaving the PD to handle only the residual.

Without that, the aircraft physically cannot be held at the incidence it
needs, and quietly descends while the controller commands full nose-up.

### The back-transition tilts first

Going forward, tilt is scheduled on airspeed — the rotors only tilt further
once the wing has enough flow to take up the slack, which makes the
transition self-pacing.

Coming back, tilt goes straight to hover and the actuator's slew rate paces
it. Scheduling *that* on airspeed would deadlock: the rotors would wait for
the aircraft to slow down, but with the rotors still pointing forward the
only way to decelerate is to pitch up, which stalls the wing and drops the
aircraft out of the sky.

## Measured mission

Hover at 25 m → transition → cruise → back-transition → hover, `dt = 0.01`:

```text
   t             mode     alt     Va   tilt   alpha  wing%
 0.0            hover   25.00   0.00    0.0    0.00    0.0
20.0           cruise   23.21  25.80   55.8    2.34  111.2
40.0           cruise   25.01  24.01   90.0    2.48   99.2
110.0 back_transition   25.06  23.99   89.8    2.48   99.1
120.0           hover   24.94   2.56    0.0    7.17    2.3
170.0           hover   25.00   0.00    0.0    0.00    0.0
```

Worst-case altitude excursion over the whole mission is 6.6 m, during the
initial acceleration; cruise settles to within 0.06 m.

::: warning The back-transition is a high-alpha manoeuvre
The aircraft spends roughly 3 seconds past the stall boundary while
decelerating. That is physically what a real VTOL does when it flares to a
stop, and the rotors are carrying it by then — but if you are using this
model to study stall, be aware that the deceleration deliberately enters it.
:::

## See also

- [Fixed-wing](/vehicles/fixed-wing) — the shared wing model
- [VTOL transition simulation](/simulations/vehicles/vtol-transition)
