# Erwin Lejeune - 2026-02-16
"""Full 6DOF quadrotor rigid-body dynamics with Newton-Euler equations.

A quadrotor is the four-rotor case of
:class:`~flybots.vehicles.multirotor.multirotor.Multirotor`, and this module
is a preset over it: the equations of motion, the motor lag and the mixer
all live upstairs, and what is left here is the parameter set that says
"four rotors, one arm length, one frame letter". The mixing matrix that
used to be written out by hand for ``x`` and ``+`` frames now falls out of
the rotor geometry, which reproduces both literals to machine precision.

Reference: R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles:
Modelling, Estimation and Control of Quadrotor," IEEE RAM, 2012.
DOI: 10.1109/MRA.2012.2206474
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from flybots.vehicles.components.allocation import Rotor
from flybots.vehicles.components.mixer import Mixer
from flybots.vehicles.multirotor.multirotor import Multirotor

__all__ = ["Quadrotor", "QuadrotorParams"]


@dataclass
class QuadrotorParams:
    """Physical parameters of a quadrotor.

    Defaults represent a 250mm-class racing/inspection quad (~1.5 kg) with
    5-inch propellers — a reasonable match for 30 m-scale simulation worlds.

    A quadrotor's geometry is fully described by an arm length and a frame
    letter, so :attr:`rotors` derives the layout rather than storing it.
    Anything that needs an explicit layout — a hexacopter, an H frame, a
    coaxial stack — uses
    :class:`~flybots.vehicles.multirotor.multirotor.MultirotorParams`
    instead.
    """

    mass: float = 1.5
    arm_length: float = 0.175
    inertia: NDArray[np.floating] = field(
        default_factory=lambda: np.diag([0.0082, 0.0082, 0.0148])
    )
    k_thrust: float = 8.55e-6
    k_torque: float = 1.36e-7
    motor_tau: float = 0.02
    omega_max: float = 1100.0
    drag_coeff: float = 0.1
    gravity: float = 9.81
    frame: str = "x"
    saturation: str = "clip"
    name: str = "quadrotor"

    @classmethod
    def crazyflie(cls) -> "QuadrotorParams":
        """Crazyflie 2.1 nano-quadrotor (27 g, 46 mm arms)."""
        return cls(
            mass=0.027,
            arm_length=0.046,
            inertia=np.diag([1.66e-5, 1.66e-5, 2.96e-5]),
            k_thrust=2.55e-8,
            k_torque=7.94e-10,
            motor_tau=0.02,
            omega_max=2500.0,
            drag_coeff=0.01,
        )

    @property
    def rotors(self) -> list[Rotor]:
        """Rotor layout derived from the arm length and frame letter."""
        return Mixer.frame_layout(self.frame, self.arm_length)

    @property
    def n_rotors(self) -> int:
        return 4

    @property
    def max_rotor_thrust(self) -> float:
        """Thrust one rotor makes at ``omega_max`` [N]."""
        return float(self.k_thrust * self.omega_max**2)

    @property
    def thrust_to_weight(self) -> float:
        """Ratio of the total available thrust to the aircraft's weight."""
        return float(4.0 * self.max_rotor_thrust / (self.mass * self.gravity))


class Quadrotor(Multirotor):
    """6DOF quadrotor simulation with motor dynamics.

    State vector (12 elements):
        ``[x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]``

    - Positions and velocities in world (ENU) frame.
    - Euler angles in ZYX convention: yaw(psi), pitch(theta), roll(phi).
    - Angular rates in body frame.

    The rotor order is **rear-left, rear-right, front-right, front-left**
    for an ``x`` frame and rear, right, front, left for a ``+`` frame, with
    spin directions ``CCW, CW, CCW, CW`` in both cases. That is the order
    the historical hard-coded mixer used, recovered from its matrix rather
    than from its comment, which had it wrong.
    """

    def __init__(self, params: QuadrotorParams | None = None) -> None:
        super().__init__(params or QuadrotorParams())

    def _build_mixer(self) -> Mixer:
        """Build the four-rotor mixer, keeping ``frame`` and ``arm_length`` on it."""
        p = self.params
        return Mixer(
            arm_length=p.arm_length,
            k_thrust=p.k_thrust,
            k_torque=p.k_torque,
            frame=p.frame,
            max_thrust=p.max_rotor_thrust,
            saturation=p.saturation,
        )
