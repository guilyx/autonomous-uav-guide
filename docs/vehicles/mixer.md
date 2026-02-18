# Control Allocation Mixer

## Theory

The mixer maps body-frame wrench $[T, \tau_x, \tau_y, \tau_z]^\top$ to individual motor forces $[f_1, f_2, f_3, f_4]^\top$. For an X-configuration:

$$\begin{bmatrix}f_1\\f_2\\f_3\\f_4\end{bmatrix} = M^{-1} \begin{bmatrix}T\\\tau_x\\\tau_y\\\tau_z\end{bmatrix}$$

Both X and +-frame configurations are supported.

## Reference

R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles," IEEE RAM, 2012. [DOI: 10.1109/MRA.2012.2206474](https://doi.org/10.1109/MRA.2012.2206474)

## API

```python
from uav_sim.vehicles.multirotor.mixer import Mixer
mixer = Mixer(arm_length=0.0397, k_thrust=1e-8, k_torque=1e-10, config="x")
motor_forces = mixer.mix(wrench)
```
