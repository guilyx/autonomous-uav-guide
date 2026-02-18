# Erwin Lejeune - 2026-02-16
"""Full 6DOF quadrotor rigid-body dynamics with Newton-Euler equations.

Reference: R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles:
Modelling, Estimation and Control of Quadrotor," IEEE RAM, 2012.
DOI: 10.1109/MRA.2012.2206474
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from uav_sim.vehicles.components.mixer import Mixer
from uav_sim.vehicles.components.motor import Motor


@dataclass
class QuadrotorParams:
    """Physical parameters of a quadrotor (Crazyflie-like defaults)."""

    mass: float = 0.027
    arm_length: float = 0.046
    inertia: NDArray[np.floating] = field(
        default_factory=lambda: np.diag([1.66e-5, 1.66e-5, 2.96e-5])
    )
    k_thrust: float = 2.55e-8
    k_torque: float = 7.94e-10
    motor_tau: float = 0.02
    omega_max: float = 2500.0
    drag_coeff: float = 0.01
    gravity: float = 9.81
    frame: str = "x"


class Quadrotor:
    """6DOF quadrotor simulation with motor dynamics.

    State vector (12 elements):
        ``[x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]``

    - Positions and velocities in world (ENU) frame.
    - Euler angles in ZYX convention: yaw(psi), pitch(theta), roll(phi).
    - Angular rates in body frame.
    """

    STATE_SIZE = 12

    # Named indices for clarity.
    IX, IY, IZ = 0, 1, 2
    IPHI, ITHETA, IPSI = 3, 4, 5
    IVX, IVY, IVZ = 6, 7, 8
    IP, IQ, IR = 9, 10, 11

    def __init__(self, params: QuadrotorParams | None = None) -> None:
        self.params = params or QuadrotorParams()
        p = self.params

        self.mixer = Mixer(
            arm_length=p.arm_length,
            k_thrust=p.k_thrust,
            k_torque=p.k_torque,
            frame=p.frame,
        )

        directions = [-1, 1, -1, 1] if p.frame == "x" else [-1, 1, -1, 1]
        self.motors = [
            Motor(
                k_thrust=p.k_thrust,
                k_torque=p.k_torque,
                tau=p.motor_tau,
                omega_max=p.omega_max,
                direction=d,
            )
            for d in directions
        ]

        self.state = np.zeros(self.STATE_SIZE)
        self.time = 0.0

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def position(self) -> NDArray[np.floating]:
        return self.state[:3].copy()

    @property
    def euler(self) -> NDArray[np.floating]:
        return self.state[3:6].copy()

    @property
    def velocity(self) -> NDArray[np.floating]:
        return self.state[6:9].copy()

    @property
    def angular_velocity(self) -> NDArray[np.floating]:
        return self.state[9:12].copy()

    def reset(
        self,
        position: NDArray[np.floating] | None = None,
        euler: NDArray[np.floating] | None = None,
        velocity: NDArray[np.floating] | None = None,
        angular_velocity: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Reset quadrotor to a given state. Returns the new state."""
        self.state = np.zeros(self.STATE_SIZE)
        if position is not None:
            self.state[:3] = position
        if euler is not None:
            self.state[3:6] = euler
        if velocity is not None:
            self.state[6:9] = velocity
        if angular_velocity is not None:
            self.state[9:12] = angular_velocity
        for m in self.motors:
            m.reset(0.0)
        self.time = 0.0
        return self.state.copy()

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def rotation_matrix(phi: float, theta: float, psi: float) -> NDArray[np.floating]:
        """ZYX Euler angles → rotation matrix (body → world)."""
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)
        return np.array(
            [
                [cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp],
                [sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp],
                [-st, ct * sp, ct * cp],
            ]
        )

    # ------------------------------------------------------------------
    # Equations of motion
    # ------------------------------------------------------------------

    def _derivatives(self, state: NDArray[np.floating], wrench: NDArray[np.floating]):
        """Compute state derivative given current state and body wrench."""
        p = self.params
        _, _, _, phi, theta, _psi, vx, vy, vz, bp, bq, br = state

        T, tau_x, tau_y, tau_z = wrench
        R = self.rotation_matrix(phi, theta, _psi)
        inertia = p.inertia
        omega_b = np.array([bp, bq, br])

        # Translational dynamics (world frame).
        thrust_world = R @ np.array([0.0, 0.0, T / p.mass])
        gravity = np.array([0.0, 0.0, -p.gravity])
        drag = -p.drag_coeff * np.array([vx, vy, vz]) / p.mass
        acc = thrust_world + gravity + drag

        # Rotational dynamics (body frame).
        tau_b = np.array([tau_x, tau_y, tau_z])
        omega_dot = np.linalg.solve(inertia, tau_b - np.cross(omega_b, inertia @ omega_b))

        # Euler angle kinematics.
        cp, sp = np.cos(phi), np.sin(phi)
        ct = np.cos(theta)
        if abs(ct) < 1e-10:
            ct = 1e-10  # avoid gimbal lock singularity

        euler_dot = np.array(
            [
                bp
                + sp * (bq * np.sin(theta) + br * np.cos(theta)) / ct
                - cp * (-bq * np.cos(theta) + br * np.sin(theta)) / ct
                + sp * bq * np.sin(theta) / ct,
                bq * cp - br * sp,
                (bq * sp + br * cp) / ct,
            ]
        )
        # Simplified ZYX euler rate:
        euler_dot = np.array(
            [
                bp + (bq * sp + br * cp) * np.tan(theta),
                bq * cp - br * sp,
                (bq * sp + br * cp) / ct,
            ]
        )

        dstate = np.zeros(self.STATE_SIZE)
        dstate[:3] = np.array([vx, vy, vz])
        dstate[3:6] = euler_dot
        dstate[6:9] = acc
        dstate[9:12] = omega_dot
        return dstate

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, wrench: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Advance simulation by ``dt`` seconds using RK4 integration.

        Args:
            wrench: ``[T, tau_x, tau_y, tau_z]`` body-frame wrench.
            dt: Time step [s].

        Returns:
            New state vector (12,).
        """
        wrench = np.asarray(wrench, dtype=np.float64)

        # Update motor dynamics.
        forces = self.mixer.wrench_to_forces(wrench)
        for i, motor in enumerate(self.motors):
            omega_cmd = motor.thrust_to_omega(forces[i])
            motor.step(omega_cmd, dt)

        # Actual wrench from motor states.
        actual_forces = np.array([m.thrust for m in self.motors])
        actual_wrench = self.mixer.forces_to_wrench(actual_forces)

        # RK4 integration.
        k1 = self._derivatives(self.state, actual_wrench)
        k2 = self._derivatives(self.state + 0.5 * dt * k1, actual_wrench)
        k3 = self._derivatives(self.state + 0.5 * dt * k2, actual_wrench)
        k4 = self._derivatives(self.state + dt * k3, actual_wrench)
        self.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Safety: clamp angular rates to prevent numerical blow-up.
        _OMEGA_MAX = 50.0
        self.state[9:12] = np.clip(self.state[9:12], -_OMEGA_MAX, _OMEGA_MAX)

        # Safety: if NaN crept in, freeze state at last valid value.
        if np.any(np.isnan(self.state)):
            self.state = np.nan_to_num(self.state, nan=0.0)

        # Normalise angles to [-pi, pi].
        self.state[3:6] = (self.state[3:6] + np.pi) % (2 * np.pi) - np.pi

        # Prevent sinking below ground plane.
        if self.state[2] < 0.0:
            self.state[2] = 0.0
            self.state[8] = max(self.state[8], 0.0)

        self.time += dt
        return self.state.copy()

    def hover_wrench(self) -> NDArray[np.floating]:
        """Return the wrench ``[T, 0, 0, 0]`` for hover at current mass."""
        return np.array([self.params.mass * self.params.gravity, 0.0, 0.0, 0.0])

    def get_motor_speeds(self) -> NDArray[np.floating]:
        """Return current motor angular velocities [rad/s]."""
        return np.array([m.omega for m in self.motors])
