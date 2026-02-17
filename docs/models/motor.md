# Motor Model

## Theory

Each motor is modelled as a first-order system with time constant $\tau_m$:

$$\dot{\omega} = \frac{1}{\tau_m}(\omega_{\text{cmd}} - \omega)$$

Thrust and torque are related to angular velocity via:

$$T = k_T \omega^2, \quad Q = k_Q \omega^2$$

## Reference

R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles," IEEE RAM, 2012. [DOI: 10.1109/MRA.2012.2206474](https://doi.org/10.1109/MRA.2012.2206474)

## API

```python
from quadrotor_sim.models.motor import Motor

motor = Motor()
motor.step(omega_cmd, dt)
thrust = motor.thrust
```
