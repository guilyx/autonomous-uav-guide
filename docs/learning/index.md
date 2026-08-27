# Reinforcement learning

Six control tasks built on the same physics as the rest of the library,
plus a trainer that needs nothing but NumPy.

```bash
flybots envs                # list the tasks
flybots train hover         # learn to hold position from scratch
flybots play hover --policy policies/hover.npz --gif hover.gif
```

<CardGrid :items="[
  { title: 'hover', meta: 'quadrotor', pill: 'easy', link: '/learning/environments#hover',
    body: 'Hold a point in space from a randomised offset, tilt and velocity.' },
  { title: 'waypoint', meta: 'quadrotor', pill: 'medium', link: '/learning/environments#waypoint',
    body: 'Fly to a random goal and settle on it, not just pass through it.' },
  { title: 'landing', meta: 'quadrotor', pill: 'medium', link: '/learning/environments#landing',
    body: 'Descend and touch down gently inside a pad. Reward scales with how soft.' },
  { title: 'trajectory', meta: 'quadrotor', pill: 'hard', link: '/learning/environments#trajectory',
    body: 'Track a moving Lissajous reference. The target never stops.' },
  { title: 'fw-cruise', meta: 'fixed-wing', pill: 'medium', link: '/learning/environments#fw-cruise',
    body: 'Hold altitude, airspeed and course on a wing that stalls if you slow down.' },
  { title: 'fw-waypoint', meta: 'fixed-wing', pill: 'hard', link: '/learning/environments#fw-waypoint',
    body: 'Fly a route of four waypoints without stalling or losing the aircraft.' }
]" />

## Why a gym here

Most drone RL benchmarks either wrap a heavyweight simulator you cannot
read, or use dynamics simple enough that what the policy learns does not
transfer to anything. These tasks sit on the same 6DOF models the rest of
the library uses — the quadrotor with its motor lag, the fixed wing with
its stall — so the control problem is the real one, and you can read every
line of the thing you are learning against.

The fixed-wing tasks are the ones worth your time. A wing cannot stop, it
stalls if it slows down, and altitude, airspeed and heading are all coupled
through four surfaces. That is a much less forgiving test of a learned
controller than hovering.

## The API

Gymnasium's interface, without the Gymnasium dependency:

```python
from flybots.gym import make

env = make("hover", seed=0)
observation, info = env.reset()

for _ in range(500):
    action = env.action_space.sample(env.rng)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

If Gymnasium *is* installed, the environments register themselves on import
and work with any standard RL library:

```python
import gymnasium as gym
import flybots.gym          # registers flybots/Hover-v0 and friends

env = gym.make("flybots/Hover-v0")
```

```bash
pip install "flybots[gym]"
```

## Training

```python
from flybots.gym import train, evaluate

result = train("hover", iterations=100, seed=0)
print(result.best_return)

summary = evaluate("hover", result.policy, episodes=25)
print(summary)
# {'mean_return': ..., 'mean_length': ..., 'pct_ground_contact': ...}
```

The default optimiser is Augmented Random Search over a **linear** policy —
about 80 parameters. On body-frame errors a linear policy is structurally a
PD controller, which is the shape these tasks want.

`evaluate` reports the share of episodes ending in each termination reason
alongside the return, because *how* a policy fails tells you far more than
the number.

See [Training a policy](/learning/training) for how it works and the four
choices that decide whether it learns at all.

## Design notes

Several of the decisions in these environments are the difference between
a task that trains in minutes and one that never leaves the ground:

- Actions are centred on **equilibrium**, so a zero action hovers or flies
  level rather than dropping the aircraft.
- Observations are **body-frame**, matching the frame the action acts in.
- Attitude enters as a **rotation matrix**, not Euler angles.
- Every candidate is scored on the **same episodes**.

[Designing a task](/learning/design-notes) works through why each of those
matters, with the measurements that motivated them.

## Honest limits

- The trainer is derivative-free. It solves these tasks; it is not a
  substitute for PPO or SAC on a hard, high-dimensional problem.
- There is no domain randomisation over vehicle parameters, so a policy
  learned here is tuned to one airframe.
- There is no wind, so a policy never learns to reject it.
- Nothing here is sim-to-real. The models omit ground effect, rotor-wake
  interaction and aeroelasticity — see
  [what the models do not include](/vehicles/#what-they-deliberately-do-not-model).
