# Frames and conventions

Read this page before writing a controller. Every sign error in UAV code
traces back to a frame assumption someone did not write down, and this
library makes several choices that differ from the aerodynamics textbooks
it implements.

## The two frames

| | Frame | Axes |
|---|---|---|
| World | **ENU** | `x` east, `y` north, `z` **up** |
| Body | **FLU** | `x` forward, `y` **left**, `z` **up** |

Attitude is ZYX Euler `(roll φ, pitch θ, yaw ψ)`, and
`uav_sim.frames.transforms.euler_to_rotation` returns the body → world
matrix.

```python
from uav_sim.frames.transforms import euler_to_rotation

R = euler_to_rotation(roll, pitch, yaw)
velocity_world = R @ velocity_body
velocity_body = R.T @ velocity_world
```

## Positive pitch is nose-down

This is the one that catches people.

In a Forward-Left-Up frame, the body `y` axis points **left**. A positive
rotation about it follows the right-hand rule: thumb along `+y` (left),
fingers curl from `+z` (up) toward `+x` (forward). That is the nose going
**down**.

Aerospace texts use Forward-Right-Down, where `y` points right and the same
positive rotation raises the nose. So:

| | This library (FLU) | Textbooks (FRD) |
|---|---|---|
| `θ > 0` | nose **down** | nose **up** |
| `φ > 0` | bank **right** | bank **right** |
| `ψ` increases | **counter-clockwise** (ENU) | clockwise (NED) |

Roll is the only one that agrees, because flipping *both* `y` and `z`
preserves the sense of a rotation about `x`.

::: warning Banking right decreases yaw
Because the world frame is ENU, heading increases counter-clockwise. An
aircraft that banks right turns clockwise, so `ψ` goes **down**. A course
controller that does not account for this will fly a stable, confident
circle in the wrong direction.
:::

For display, the fixed-wing model exposes `pitch_up`, which is simply
`-theta`:

```python
aircraft.state[4]      # 0.15  — stored FLU pitch, nose-down positive
aircraft.pitch_up      # -0.15 — aerospace convention, nose-up positive
```

## Porting textbook equations

The aerodynamic model in
`uav_sim/vehicles/fixed_wing/aerodynamics.py` is written in **FRD**,
exactly as Beard & McLain state it. Rather than rewriting thirty
coefficient equations with flipped signs — which is how transcription
errors get in — the vehicle converts at the boundary:

```python
from uav_sim.frames.transforms import flu_to_frd, frd_to_flu

wrench = aero_wrench(
    velocity_body_frd=flu_to_frd(velocity_flu),
    rates_body_frd=flu_to_frd(rates_flu),
    ...
)
force_flu = frd_to_flu(wrench.force)
moment_flu = frd_to_flu(wrench.moment)
```

FLU and FRD differ by a 180° rotation about the shared `x` axis, so both
conversions are the same sign flip on `y` and `z`, and each is its own
inverse. Note this also flips pitch- and yaw-like rates: a `q` that is
nose-up in FRD is nose-down in FLU.

Do the same for any equation you port. Convert at the edge; keep the
citation's algebra intact.

## State vectors

All three airframes use a 12-element state, but **the velocity block
differs**, and this is deliberate rather than an oversight.

```text
Quadrotor / Tiltrotor:  [x y z  φ θ ψ  vx vy vz  p q r]
                                        └── world ENU ──┘

Fixed-wing:             [x y z  φ θ ψ  u  v  w   p q r]
                                        └── body FLU ──┘
```

A multirotor is commanded in world-frame velocity, so storing velocity
there keeps its controllers direct. A fixed wing's aerodynamics depend on
the *air-relative body* velocity — angle of attack is
`atan2(-w, u)` — so storing it in the body frame avoids rotating it back
and forth on every dynamics evaluation.

Both expose the other form as a property:

```python
quad.velocity            # world ENU
aircraft.body_velocity   # [u, v, w] in FLU
aircraft.velocity        # world ENU, computed as R @ [u, v, w]
```

Angular rates are body-frame in every model.

## Units

SI throughout, radians for angles, everywhere in the API. Degrees appear
only inside display strings and plot labels.

| Quantity | Unit |
|---|---|
| Position, altitude | m |
| Velocity, airspeed | m/s |
| Angles, angle of attack, sideslip | rad |
| Angular rates | rad/s |
| Force, thrust, lift, drag | N |
| Moment, torque | N·m |
| Mass | kg |
| Air density | kg/m³ |
| Wing area | m² |
| Throttle | dimensionless, `[0, 1]` |
| Control surfaces | rad, clamped to ±30° |

## Aerodynamic angles

```python
aircraft.alpha   # angle of attack  [rad], positive nose-up into the airflow
aircraft.beta    # sideslip         [rad], positive with wind from the right
```

Both follow the **aerodynamic** convention even though the state is stored
in FLU, because that is how every reference defines them and how every
coefficient is tabulated. The model negates internally; you do not need
to.

Angle of attack is measured from the **body-relative** airflow, so pitch
attitude changes it. In level flight at 20 m/s, an aircraft held 10°
nose-up has `alpha ≈ 10°` — not zero.

## Gravity

World gravity is `[0, 0, -g]`. In the body frame:

```python
from uav_sim.frames.transforms import gravity_in_body

gravity_in_body(roll, pitch)   # level flight -> [0, 0, -9.81]
```

Body-down is `-z` in a Forward-Left-Up frame, so a level aircraft sees
gravity along negative body `z`. If your model has `+g` there, it will fly
upside down and look almost right for the first few seconds.

## A checklist for new models

1. Does `z` increase upward? Fly the vehicle with zero control and confirm
   altitude *decreases*.
2. Does the gravity term give `[0, 0, -g]` at level attitude?
3. Does the body velocity rotate into the world with `R @ v`, not `R.T @ v`?
4. Does a positive roll produce a turn in the direction you expect —
   remembering that ENU yaw decreases in a right turn?
5. Is every parameter in your dataclass read by the dynamics? Write the
   test that fails when you change one.

Item 5 is not pedantry. The previous fixed-wing model declared `Cm0`,
`Cma` and `e_oswald` and never read any of them, which left the aircraft
with no static pitch stability and no induced drag — while looking, in the
source, like a complete model.
