# Flight models

Three airframe families, one set of conventions, one base class. They
compose because they agree on frames — see
[Frames and conventions](/guide/conventions).

<CardGrid :items="[
  {
    title: 'Quadrotor',
    meta: 'multirotor',
    link: '/vehicles/quadrotor',
    body: '6DOF Newton-Euler rigid body with per-motor first-order dynamics and a configurable mixer.'
  },
  {
    title: 'Multirotor',
    meta: 'N rotors',
    link: '/vehicles/multirotor',
    body: 'Any even rotor count and any arm geometry, with the mixing matrix derived from rotor positions and spin directions rather than hard-coded.'
  },
  {
    title: 'Fixed-wing',
    meta: 'aerodynamic',
    link: '/vehicles/fixed-wing',
    body: 'Full Beard & McLain coefficient build-up: stability derivatives, induced drag, post-stall lift, momentum-theory propeller.'
  },
  {
    title: 'VTOL tilt-rotor',
    meta: 'hybrid',
    link: '/vehicles/vtol',
    body: 'Rotors that tilt from hover to cruise, sharing the fixed-wing airframe model, with a rate-limited tilt actuator.'
  }
]" />

## What each one models

| | Quadrotor | Fixed-wing | Tilt-rotor |
|---|---|---|---|
| Rigid-body 6DOF | yes | yes | yes |
| Integrator | RK4 | RK4 | RK4 |
| Actuator dynamics | per-motor first order | surface rate limits | tilt slew limit |
| Aerodynamics | linear body drag | full coefficient build-up | shared wing model |
| Stall | — | yes, flat-plate blend | yes, shared |
| Propulsion | thrust-coefficient motors | momentum-theory propeller | tilting rotor thrust |
| Trim solver | hover wrench (closed form) | numerical, `compute_trim` | — |
| Presets | 6 (4 quad, hex, coaxial octo) | 4 | 1 |
| State velocity | world ENU | body FLU | world ENU |
| Control input | `[T, τx, τy, τz]` | `[δe, δa, δr, δt]` | `[T, τx, τy, τz, tilt]` |

## What they deliberately do not model

Being explicit about this matters more than the feature list. None of the
models include:

- **Wind, gusts or turbulence.** Airspeed equals ground speed everywhere.
  Adding a wind field is a natural extension; the aerodynamics already work
  from an air-relative velocity, so it would enter in one place.
- **Ground effect**, rotor-wake interaction, or blade flapping.
- **Compressibility or Reynolds-number effects.** Coefficients are constant.
- **Structural flexibility.** Every airframe is rigid.
- **Actuator failure or saturation dynamics** beyond simple rate and range
  limits.
- **Sensor mounting on the dynamics.** Sensors read the true state and add
  their own noise models; they do not load the airframe.

The models are built to be *correct within their stated scope* and readable,
not to be a certified flight-dynamics package. See the
[safety note](https://github.com/guilyx/flybots/blob/main/SECURITY.md#a-note-on-flight-safety).

## The shared base

Every airframe subclasses `UAVBase`, which owns the state vector, RK4
integration and reset semantics:

```python
from flybots.vehicles.base import UAVBase, UAVParams

class MyAirframe(UAVBase):
    @property
    def state_dim(self) -> int: ...

    @property
    def control_dim(self) -> int: ...

    def _dynamics(self, state, control):
        """Return dx/dt."""
```

`Quadrotor` predates the base class and keeps its own `step()` so that
motor states integrate alongside the rigid body; it follows the same state
layout and conventions.

## Choosing one

- **Learning control theory, or anything indoors** → quadrotor. It hovers,
  which makes every experiment cheap to set up.
- **Range, endurance, or aerodynamics** → fixed-wing. It cannot hover, so
  it forces you to think about trim, stall and coordinated turns.
- **Transition control** → tilt-rotor. The interesting part is the handover
  of lift from rotors to wing, which is a genuinely hard control-allocation
  problem.

## Common pitfalls

::: warning Positive pitch is nose-down
A consequence of the Forward-Left-Up body frame. Use `aircraft.pitch_up`
when displaying an attitude to a human.
:::

::: warning A fixed wing needs a big world
The default Aerosonde cruises at 35 m/s and stalls near 13 m/s. It cannot
fly in the 30 m worlds the quadrotor simulations use — it will leave the
volume in under a second. Use `MINI_TRAINER` for small worlds, or scale the
world up.
:::

::: warning Quadrotor motors start stopped
`reset()` zeroes the motor speeds. Commanding hover thrust to stopped
motors still drops the aircraft while they spin up. Pre-load them with
`motor.reset(motor.thrust_to_omega(hover / 4))` when you want an
undisturbed hover.
:::
