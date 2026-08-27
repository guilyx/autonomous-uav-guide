# Erwin Lejeune - 2026-02-17
"""IMU sensor model with bias, noise, and saturation.

Reference: N. Trawny, S. I. Roumeliotis, "Indirect Kalman Filter for 3D
Attitude Estimation," TR, 2005.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flybots.sensors.base import Sensor
from flybots.vehicles.multirotor.quadrotor import Quadrotor


class IMU(Sensor):
    """6-axis IMU: 3-axis accelerometer + 3-axis gyroscope.

    Returns a 6-vector: ``[ax, ay, az, gx, gy, gz]``.

    The accelerometer reports **specific force in the body frame**, i.e.
    ``R(q)ᵀ (a_world + g ẑ)`` — so a stationary vehicle reads ``+g`` on
    its body z-axis, not zero.  World-frame acceleration is either passed
    in explicitly or finite-differenced from the velocity in the state
    between successive calls.

    Both channels carry a constant turn-on bias drawn at construction on
    top of white noise.  The bias is what makes dead-reckoning diverge
    quadratically; white noise alone only produces a slow random walk.

    Parameters
    ----------
    accel_noise_std : accelerometer white noise [m/s²].
    gyro_noise_std : gyroscope white noise [rad/s].
    accel_bias_std : spread of the constant accelerometer bias [m/s²].
    gyro_bias_std : spread of the constant gyroscope bias [rad/s].
    rate_hz : measurement rate, also the default finite-difference step.
    gravity : gravitational acceleration [m/s²].
    seed : random seed for reproducible noise.
    """

    def __init__(
        self,
        accel_noise_std: float = 0.1,
        gyro_noise_std: float = 0.01,
        accel_bias_std: float = 0.005,
        gyro_bias_std: float = 0.001,
        rate_hz: float = 200.0,
        gravity: float = 9.81,
        seed: int | None = None,
    ) -> None:
        super().__init__(rate_hz, seed)
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.gravity = gravity
        self._accel_bias = self._rng.normal(0, accel_bias_std, 3)
        self._gyro_bias = self._rng.normal(0, gyro_bias_std, 3)
        self._prev_vel: NDArray[np.floating] | None = None

    @property
    def accel_bias(self) -> NDArray[np.floating]:
        """Constant accelerometer turn-on bias [m/s²]."""
        return self._accel_bias.copy()

    @property
    def gyro_bias(self) -> NDArray[np.floating]:
        """Constant gyroscope turn-on bias [rad/s]."""
        return self._gyro_bias.copy()

    def reset(self) -> None:
        """Forget the previous velocity used for finite differencing."""
        self._prev_vel = None

    def sense(
        self,
        state: NDArray[np.floating],
        world=None,
        accel_world: NDArray[np.floating] | None = None,
        dt: float | None = None,
    ) -> NDArray[np.floating]:
        """Return ``[ax, ay, az, gx, gy, gz]`` for the given vehicle *state*.

        Parameters
        ----------
        state : 12-element ``[x,y,z,φ,θ,ψ,vx,vy,vz,p,q,r]``.
        world : unused, kept for the :class:`Sensor` interface.
        accel_world : true world-frame acceleration [m/s²].  When omitted
            it is finite-differenced from the velocity in *state*.
        dt : step used for the finite difference (defaults to ``1/rate_hz``).
        """
        vel = state[6:9] if len(state) >= 9 else np.zeros(3)
        gyro_true = state[9:12] if len(state) >= 12 else np.zeros(3)
        euler = state[3:6] if len(state) >= 6 else np.zeros(3)

        step = self.dt if dt is None else max(float(dt), 1e-9)
        if accel_world is None:
            if self._prev_vel is None:
                accel_world = np.zeros(3)
            else:
                accel_world = (vel - self._prev_vel) / step
        self._prev_vel = np.asarray(vel, dtype=float).copy()

        # Specific force: what a proof mass actually feels, in body axes.
        R = Quadrotor.rotation_matrix(*euler)
        gravity_reaction = np.array([0.0, 0.0, self.gravity])
        specific_force = R.T @ (np.asarray(accel_world, dtype=float) + gravity_reaction)

        accel = specific_force + self._accel_bias + self._rng.normal(0, self.accel_noise_std, 3)
        gyro = gyro_true + self._gyro_bias + self._rng.normal(0, self.gyro_noise_std, 3)
        self._last_measurement = np.concatenate([accel, gyro])
        return self._last_measurement
