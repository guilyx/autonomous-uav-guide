# Quadrotor

Full 6DOF Newton-Euler rigid-body dynamics with per-motor first-order
response.

> R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles: Modelling,
> Estimation and Control of Quadrotor", IEEE RAM, 2012.
> [doi:10.1109/MRA.2012.2206474](https://doi.org/10.1109/MRA.2012.2206474)

Source: [`uav_sim/vehicles/multirotor/`](https://github.com/guilyx/flybots/tree/main/src/uav_sim/vehicles/multirotor)

## Quick start

```python
import numpy as np
from uav_sim.vehicles.multirotor import Quadrotor

quad = Quadrotor()
quad.reset(position=np.array([0.0, 0.0, 2.0]))

hover = quad.hover_wrench()
for motor in quad.motors:               # motors start stopped
    motor.reset(motor.thrust_to_omega(hover[0] / 4.0))

for _ in range(500):
    quad.step(hover, 0.005)
```

## State and control

```text
state  = [x, y, z, φ, θ, ψ, vx, vy, vz, p, q, r]
wrench = [T, τx, τy, τz]
```

Position and velocity are world **ENU**; rates are body-frame. The control
input is a body wrench — total thrust along body `+z` plus three moments.

## Actuation chain

The wrench is not applied directly. It goes through the mixer and the
motors, so the aircraft feels the thrust the motors can actually produce:

```text
wrench -> mixer -> per-motor thrust -> ω command
                                        │
                              first-order motor lag (τ)
                                        │
                        actual ω -> actual thrust -> actual wrench
```

```python
quad.get_motor_speeds()     # rad/s, per motor
quad.params.motor_tau       # motor time constant [s]
quad.params.omega_max       # saturation
```

This is why an instantaneous step in commanded thrust does not produce an
instantaneous change in acceleration, and why aggressive controllers can
excite the motor dynamics.

### Mixer

`Mixer` maps between wrench and per-motor thrust for `x` and `+` frames,
using the arm length and the thrust/torque coefficients. Yaw torque comes
from the reaction torque of the counter-rotating pairs.

## Presets

```python
from uav_sim.vehicles import VehiclePreset, create_quadrotor

quad = create_quadrotor(VehiclePreset.CRAZYFLIE)      # 27 g nano
quad = create_quadrotor(VehiclePreset.DJI_MINI)       # 249 g
quad = create_quadrotor(VehiclePreset.RACING_250)     # 1.5 kg (default)
quad = create_quadrotor(VehiclePreset.DJI_MATRICE)    # 3.6 kg
quad = create_quadrotor(VehiclePreset.RACING_250, mass=2.0)   # override
```

| Preset | Mass | Arm | ω max |
|---|---|---|---|
| `CRAZYFLIE` | 0.027 kg | 46 mm | 2500 rad/s |
| `DJI_MINI` | 0.249 kg | 110 mm | 1800 rad/s |
| `RACING_250` | 1.5 kg | 175 mm | 1100 rad/s |
| `DJI_MATRICE` | 3.6 kg | 320 mm | 800 rad/s |

The three-order-of-magnitude mass span is a useful stress test for any
controller you write.

## Numerical guards

`step()` clamps body rates to ±50 rad/s, wraps Euler angles to `[-π, π]`,
replaces any NaN that appears, and stops the aircraft sinking below `z = 0`.
These exist so a diverging controller produces a visibly bad trajectory
rather than a wall of NaN.

## Control stack

The cascaded controllers in `uav_sim.control` layer over this model:

```text
PositionController -> VelocityController -> AttitudeController -> RateController
```

with `FlightController` composing them and `StateManager` running the
ARM → TAKEOFF → HOVER → TRACKING → LAND mode sequence.

## See also

- [Quadrotor dynamics simulation](/simulations/vehicles/quadrotor-dynamics)
- [PID hover](/simulations/path-tracking/pid-hover)
- [LQR hover](/simulations/path-tracking/lqr-hover)
