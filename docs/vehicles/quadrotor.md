# 6DOF Quadrotor Dynamics

## Theory

The quadrotor is modelled as a rigid body in SE(3) using Newton-Euler equations.  The 12-dimensional state vector is:

$$\mathbf{x} = [x, y, z, \phi, \theta, \psi, \dot{x}, \dot{y}, \dot{z}, p, q, r]^\top$$

where $(x,y,z)$ is the position in the world frame, $(\phi,\theta,\psi)$ are Euler angles (roll, pitch, yaw), $(\dot{x},\dot{y},\dot{z})$ is the translational velocity, and $(p,q,r)$ are body angular rates.

The translational dynamics follow:

$$m\ddot{\mathbf{r}} = R \begin{bmatrix}0\\0\\T\end{bmatrix} + m\begin{bmatrix}0\\0\\-g\end{bmatrix}$$

and rotational dynamics:

$$I\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times I\boldsymbol{\omega}$$

Integration uses 4th-order Runge-Kutta (RK4).

## Reference

R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles: Modelling, Estimation and Control of Quadrotor," IEEE RAM, vol. 19, no. 3, pp. 20-32, 2012. [DOI: 10.1109/MRA.2012.2206474](https://doi.org/10.1109/MRA.2012.2206474)

## API

```python
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

quad = Quadrotor()
quad.reset(position=np.array([0.0, 0.0, 0.0]))
quad.step(wrench, dt)        # wrench = [thrust, tau_x, tau_y, tau_z]
state = quad.state            # 12-dim vector
```

## See Also

- [Motor Model](motor.md)
- [Mixer](mixer.md)
