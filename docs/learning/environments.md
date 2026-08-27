# Environments

Six tasks. All follow the Gymnasium contract — `reset()` returns
`(obs, info)`, `step()` returns
`(obs, reward, terminated, truncated, info)`.

```bash
flybots envs
```

| id | vehicle | level | obs | act |
|---|---|---|---|---|
| `hover` | quadrotor | easy | 18 | 4 |
| `waypoint` | quadrotor | medium | 18 | 4 |
| `landing` | quadrotor | medium | 18 | 4 |
| `trajectory` | quadrotor | hard | 21 | 4 |
| `fw-cruise` | fixed-wing | medium | 16 | 4 |
| `fw-waypoint` | fixed-wing | hard | 16 | 4 |

## Quadrotor tasks

All four share an action space and an observation layout.

**Action** — normalised body wrench in `[-1, 1]⁴`:

```text
[thrust, tau_x, tau_y, tau_z]
```

Thrust is centred on hover, so `0` is exactly the thrust that balances
weight. Torque limits are derived from the airframe's inertia and a
configured peak angular acceleration, so the same numbers give comparable
handling on a 27 g Crazyflie and a 3.6 kg Matrice.

**Observation** — 18 elements, body-frame:

```text
[0:3]    position error, body frame
[3:6]    velocity, body frame
[6:15]   rotation matrix, flattened
[15:18]  body angular rates
```

See [Designing a task](/learning/design-notes) for why the frame and the
rotation-matrix encoding matter.

### hover {#hover}

Hold `[0, 0, 3]`. The quadrotor spawns displaced by around a metre, tilted,
rotated to a random yaw, and moving.

Reward is a Gaussian in distance plus small penalties on speed, tilt, spin
and control effort, with a bonus inside 0.25 m. Maximum achievable return
is roughly 750 over a 500-step episode.

Terminates on ground contact, tumbling past 75°, or leaving the volume.

### waypoint {#waypoint}

Fly to a randomly placed goal and **settle** on it. Holding station within
0.25 m and 0.3 m/s for a full second terminates the episode successfully,
so the policy learns to arrive and stop rather than to fly through at speed.

### landing {#landing}

Descend from 3–6 m and touch down inside a 0.6 m pad.

The reward rewards centring and a *controlled* descent rate — proportional
to altitude, so it slows as it approaches — and pays a touchdown bonus that
scales with how soft the landing is:

```python
bonus = 50.0 * (1.0 - speed / max_touchdown_speed)
```

Both a gentle landing and a crash end the episode, so the bonus is what
makes the difference between them matter.

### trajectory {#trajectory}

Track a moving reference on a 3-D Lissajous figure. The observation carries
three extra elements: where the reference will be in 0.5 s, in the body
frame, so the policy can lead the target rather than chase it.

Harder than `waypoint` because the target never stops.

## Fixed-wing tasks

A wing cannot stop, stalls if it slows down, and couples altitude, airspeed
and heading through the same four surfaces. These are the more interesting
tasks.

**Action** — deltas from the [trim solution](/vehicles/trim):

```text
[elevator, aileron, rudder, throttle]
```

A zero action flies straight and level, so the policy learns the
*correction*, not equilibrium flight from scratch.

**Observation** — 16 elements:

```text
[0:3]    altitude / airspeed / course errors, normalised
[3:9]    sin and cos of roll, pitch, yaw
[9:12]   body rates, scaled
[12]     angle of attack
[13]     sideslip
[14]     airspeed / cruise airspeed
[15]     stall flag
```

Course error uses **course**, not heading — see
[Autopilots](/vehicles/autopilots#the-course-loop-closes-on-course-not-heading).

Episodes start from trim and are then upset: a few degrees of roll and
pitch, a few percent of airspeed, some sideslip.

### fw-cruise {#fw-cruise}

Hold a commanded altitude, airspeed and course. The command is
**re-randomised part-way through each episode**, so the policy has to learn
to acquire a new setpoint, not to sit on the one it started at.

Terminates on ground contact, leaving the altitude band, rolling past 120°,
or staying stalled for more than two seconds. That grace period matters:
brief stalls happen in aggressive manoeuvres and should not end the episode,
but a policy that parks the aircraft in a stall should not be allowed to
keep accruing reward.

### fw-waypoint {#fw-waypoint}

Fly a route of four waypoints, laid out ahead of the aircraft with each leg
a moderate turn from the last, so the course is always reachable without a
reversal. Capturing all four ends the episode successfully.

## Configuration

```python
from flybots.gym import make
from flybots.gym.quadrotor_envs import QuadrotorEnvConfig

env = make("hover", config=QuadrotorEnvConfig(
    max_episode_seconds=20.0,
    spawn_position_noise=2.0,
    max_angular_acceleration=8.0,      # gentler, easier to learn
    world_radius=10.0,
))
```

```python
from flybots.gym.fixed_wing_envs import FixedWingEnvConfig
from flybots.vehicles.fixed_wing import FixedWingPreset

env = make("fw-cruise", config=FixedWingEnvConfig(
    preset=FixedWingPreset.AEROSONDE,
    reference_altitude=200.0,
    stall_grace_seconds=1.0,
))
```

## Writing your own

Subclass `UAVEnv` and implement five hooks; the base class owns episode
bookkeeping, action clipping, substepped integration and seeding.

```python
import numpy as np
from flybots.gym.base import UAVEnv
from flybots.gym.spaces import Box

class CircleEnv(UAVEnv):
    @property
    def observation_space(self): return Box(-np.inf, np.inf, shape=(18,))

    @property
    def action_space(self): return Box(-1.0, 1.0, shape=(4,))

    def _reset_task(self): ...
    def _observe(self): ...
    def _apply_action(self, action): ...
    def _reward(self, action): return reward, breakdown
    def _terminated(self): return done, reason

    @property
    def vehicle_position(self): return self.vehicle.state[:3]
```

Register it so the CLI can find it:

```python
from flybots.gym.registry import ENV_SPECS, EnvSpec

ENV_SPECS["circle"] = EnvSpec(
    "circle", lambda **kw: CircleEnv(**kw),
    "Fly a circle at constant altitude.", "quadrotor", "medium",
)
```

Before you tune anything, read
[Designing a task](/learning/design-notes) — the four choices there decide
whether a new task is learnable at all.
