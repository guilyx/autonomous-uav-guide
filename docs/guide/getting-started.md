# First flight

Five minutes, four things: hover a quadrotor, read its state, fly a wing,
and render a simulation.

If you have not installed it yet, see [Installation](/guide/installation).

## 1. Hover a quadrotor

```python
import numpy as np
from uav_sim.vehicles.multirotor import Quadrotor

quad = Quadrotor()
quad.reset(position=np.array([0.0, 0.0, 2.0]))

# Thrust that exactly balances weight.
hover = quad.hover_wrench()          # [T, tau_x, tau_y, tau_z]

# Spin the motors up first: they start at zero and have a time constant,
# so commanding hover thrust to stopped motors still drops the aircraft.
for motor in quad.motors:
    motor.reset(motor.thrust_to_omega(hover[0] / 4.0))

for _ in range(500):
    quad.step(hover, 0.005)

print(quad.position)     # [0. 0. 2.]
```

Open-loop hover holds because the wrench is exact. Perturb it and the
quadrotor drifts, as it should — a real one needs a controller.

## 2. Read the state

The 12-element state is the same shape for every vehicle:

```python
quad.state          # [x y z  φ θ ψ  vx vy vz  p q r]
quad.position       # [x, y, z]        world ENU
quad.euler          # [roll, pitch, yaw]
quad.velocity       # world ENU
quad.angular_velocity
```

`z` is **altitude**, increasing upward, and positive pitch is
**nose-down**. Both are consequences of the ENU/FLU convention — see
[Frames and conventions](/guide/conventions), which is worth ten minutes
before you write a controller.

## 3. Fly it under closed-loop control

```python
import numpy as np
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.path_tracking.pid_controller import CascadedPIDController

quad = Quadrotor()
quad.reset(position=np.array([0.0, 0.0, 1.0]))
controller = CascadedPIDController()

target = np.array([3.0, 2.0, 4.0])
for _ in range(3000):
    wrench = controller.compute(quad.state, target, dt=0.005)
    quad.step(wrench, 0.005)

print(np.round(quad.position, 2))    # close to [3. 2. 4.]
```

## 4. Fly a fixed wing

A wing cannot hover, so the interesting starting point is not "at rest" but
**trimmed** — the attitude and control setting at which steady flight is an
equilibrium. The library solves for it:

```python
from uav_sim.vehicles.fixed_wing import create_fixed_wing, FixedWingPreset

aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8)
controls = aircraft.reset_trimmed(airspeed=18.0, altitude=120.0)

print(controls)              # [elevator, aileron, rudder, throttle]

for _ in range(6000):
    aircraft.step(controls, 0.005)

print(aircraft.state[2])     # still 120.0 — trim is a real equilibrium
print(aircraft.airspeed)     # still 18.0
```

Thirty seconds of open-loop flight with no altitude drift is the strongest
single statement that the aerodynamics are self-consistent. Add a
controller to go somewhere:

```python
from uav_sim.control.fixed_wing_autopilot import FixedWingAutopilot, AutopilotCommand

pilot = FixedWingAutopilot(aircraft.fw_params)
command = AutopilotCommand(altitude=160.0, airspeed=20.0, course=1.0)

for _ in range(12_000):
    aircraft.step(pilot.compute(aircraft.state, command, 0.01), 0.01)
```

The autopilot derives its own gains from the airframe's control
derivatives, so the same code flies a 0.6 kg foam trainer and a 25 kg cargo
UAV. See [Autopilots](/vehicles/autopilots).

## 5. Run a simulation

Forty-odd simulations render an animated three-panel view:

```bash
flybots list                       # browse them
flybots info astar_3d              # references, module path, command
flybots run astar_3d               # render the GIF
```

Or as a module:

```bash
python -m uav_sim.simulations.path_planning.astar_3d
```

Each writes its GIF next to its `run.py`, plus a JSON log of the run.

## 6. Teach one to fly itself

```bash
flybots envs                       # the six RL tasks
flybots train hover                # train a policy from scratch
flybots play hover --policy policies/hover.npz --gif hover.gif
```

The trainer is pure NumPy — no deep-learning stack required. See
[Reinforcement learning](/learning/).

## Where to go next

- [Frames and conventions](/guide/conventions) — the sign rules. Read this one.
- [Flight models](/vehicles/) — what each airframe models, and what it does not.
- [CLI reference](/guide/cli) — the full command surface.
- [Algorithm atlas](/simulations/) — every simulation with a preview.
