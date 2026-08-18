# Erwin Lejeune - 2026-08-18
"""Full 6DOF multirotor rigid-body dynamics for an arbitrary rotor count.

One rigid body, one Newton-Euler integration, and a rotor layout that says
where the thrust comes from. A quadrotor, a hexacopter and a coaxial X8
differ only in that layout: the equations of motion below never count the
rotors, and the mixing matrix is derived from rotor positions and spin
directions rather than tabulated per airframe — see
:mod:`uav_sim.vehicles.components.allocation`.

The state is the same twelve elements every multirotor in this library
uses::

    [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]

with position and velocity in world **ENU**, Euler angles ZYX, and angular
rates in body **FLU**. Positive pitch is nose-down; the allocation signs
depend on it (see :doc:`/guide/conventions`).

References
----------
- R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles: Modelling,
  Estimation and Control of Quadrotor," IEEE Robotics & Automation
  Magazine, 19(3):20-32, 2012. DOI: 10.1109/MRA.2012.2206474
- M. Achtelik, K.-M. Doth, D. Gurdan, J. Stumpf, "Design of a Multi Rotor
  MAV with regard to Efficiency, Dynamics and Redundancy," AIAA Guidance,
  Navigation, and Control Conference, 2012. DOI: 10.2514/6.2012-4779
- T. A. Johansen, T. I. Fossen, "Control allocation - A survey,"
  Automatica, 49(5):1087-1103, 2013.
  DOI: 10.1016/j.automatica.2013.01.035
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from uav_sim.vehicles.components.allocation import ControlAllocation, Rotor, x_layout
from uav_sim.vehicles.components.motor import Motor

__all__ = ["Multirotor", "MultirotorParams"]

#: Body rates above this are a diverging controller, not a manoeuvre. Kept
#: at the historical quadrotor value so the guard behaves identically.
_RATE_LIMIT = 50.0


@dataclass
class MultirotorParams:
    """Physical parameters of a multirotor with an explicit rotor layout.

    The defaults describe a 550 mm-class hexacopter, which is the smallest
    airframe for which "arm length and frame letter" stops being enough of
    a description.

    Attributes:
        mass: Total mass [kg].
        inertia: Body inertia tensor about the centre of mass [kg m^2].
        rotors: Rotor positions and spin directions in body FLU.
        k_thrust: Thrust coefficient ``T = k_thrust * omega^2`` [N/(rad/s)^2].
        k_torque: Reaction torque coefficient [Nm/(rad/s)^2].
        motor_tau: First-order motor time constant [s].
        omega_max: Motor speed ceiling [rad/s].
        drag_coeff: Linear world-frame fuselage drag [N/(m/s)].
        gravity: Gravitational acceleration [m/s^2].
        saturation: How the mixer resolves an infeasible thrust request,
            ``"clip"`` or ``"prioritise_torque"``. See
            :class:`~uav_sim.vehicles.components.allocation.ControlAllocation`.
        name: Human-readable airframe name, for logs and plot titles.
    """

    mass: float = 1.8
    inertia: NDArray[np.floating] = field(default_factory=lambda: np.diag([0.025, 0.025, 0.050]))
    rotors: list[Rotor] = field(default_factory=lambda: x_layout(6, 0.275))
    k_thrust: float = 1.0e-5
    k_torque: float = 1.6e-7
    motor_tau: float = 0.03
    omega_max: float = 940.0
    drag_coeff: float = 0.12
    gravity: float = 9.81
    saturation: str = "clip"
    name: str = "multirotor"

    @property
    def n_rotors(self) -> int:
        return len(self.rotors)

    @property
    def max_rotor_thrust(self) -> float:
        """Thrust one clean-air rotor makes at ``omega_max`` [N]."""
        return float(self.k_thrust * self.omega_max**2)

    @property
    def thrust_to_weight(self) -> float:
        """Ratio of the total available thrust to the aircraft's weight.

        Accounts for each rotor's ``thrust_scale``, so a coaxial airframe
        is not credited with eight clean-air rotors when four of them are
        working in a wake.
        """
        available = self.max_rotor_thrust * sum(r.thrust_scale for r in self.rotors)
        return float(available / (self.mass * self.gravity))


class Multirotor:
    """6DOF multirotor with a derived mixer and per-rotor motor dynamics.

    State vector (12 elements)::

        [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]

    - Positions and velocities in world (ENU) frame.
    - Euler angles in ZYX convention: yaw(psi), pitch(theta), roll(phi).
    - Angular rates in body frame.

    The control input is a body wrench ``[T, tau_x, tau_y, tau_z]``,
    whatever the rotor count. That is what keeps every controller in
    :mod:`uav_sim.control` airframe-agnostic: only the mixer knows how many
    rotors there are.
    """

    STATE_SIZE = 12

    # Named indices for clarity.
    IX, IY, IZ = 0, 1, 2
    IPHI, ITHETA, IPSI = 3, 4, 5
    IVX, IVY, IVZ = 6, 7, 8
    IP, IQ, IR = 9, 10, 11

    def __init__(self, params: MultirotorParams | None = None) -> None:
        self.params = params or MultirotorParams()
        self.mixer = self._build_mixer()
        # The motors have to line up with the mixer's columns, so the layout
        # is read back off the mixer rather than out of the parameters: a
        # preset is free to describe its geometry indirectly — a frame
        # letter and an arm length — and have the mixer expand it.
        self._rotors = self.mixer.rotors
        self.motors = [
            Motor(
                # The coaxial efficiency of a rotor is a property of where
                # it sits, so it belongs to the rotor; the motor sees it as
                # a derated thrust curve.
                k_thrust=self.params.k_thrust * rotor.thrust_scale,
                k_torque=self.params.k_torque * rotor.thrust_scale,
                tau=self.params.motor_tau,
                omega_max=self.params.omega_max,
                direction=rotor.direction,
            )
            for rotor in self.rotors
        ]

        self.state = np.zeros(self.STATE_SIZE)
        self.time = 0.0

    def _build_mixer(self) -> ControlAllocation:
        """Construct the control allocation for this airframe's layout."""
        return ControlAllocation(
            self.params.rotors,
            k_thrust=self.params.k_thrust,
            k_torque=self.params.k_torque,
            max_thrust=self.params.max_rotor_thrust,
            saturation=self.params.saturation,
        )

    # ------------------------------------------------------------------
    # Airframe
    # ------------------------------------------------------------------

    @property
    def rotors(self) -> list[Rotor]:
        """The rotor layout, in mixer column order."""
        return self._rotors

    @property
    def n_rotors(self) -> int:
        return len(self.rotors)

    @property
    def rotor_positions(self) -> NDArray[np.floating]:
        """``(n, 3)`` array of rotor hub positions in body FLU [m]."""
        return np.array([r.position for r in self.rotors])

    @property
    def spin_directions(self) -> NDArray[np.floating]:
        """``(n,)`` array of ``+1`` (CCW) / ``-1`` (CW) per rotor."""
        return np.array([r.direction for r in self.rotors])

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
        """Reset the aircraft to a given state. Returns the new state."""
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

        # Euler angle kinematics (ZYX), with the cos(theta) denominator
        # floored so the result stays finite at the gimbal-lock singularity.
        cp, sp = np.cos(phi), np.sin(phi)
        ct = np.cos(theta)
        if abs(ct) < 1e-10:
            ct = 1e-10

        common = bq * sp + br * cp
        euler_dot = np.array(
            [
                bp + common * np.tan(theta),
                bq * cp - br * sp,
                common / ct,
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

        # Actual wrench from motor states. Commanding a wrench and applying
        # the one the rotors actually made is the whole point of routing
        # through the mixer: saturation and motor lag both show up here.
        actual_forces = np.array([m.thrust for m in self.motors])
        actual_wrench = self.mixer.forces_to_wrench(actual_forces)

        # RK4 integration.
        k1 = self._derivatives(self.state, actual_wrench)
        k2 = self._derivatives(self.state + 0.5 * dt * k1, actual_wrench)
        k3 = self._derivatives(self.state + 0.5 * dt * k2, actual_wrench)
        k4 = self._derivatives(self.state + dt * k3, actual_wrench)
        self.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Safety: clamp angular rates to prevent numerical blow-up.
        self.state[9:12] = np.clip(self.state[9:12], -_RATE_LIMIT, _RATE_LIMIT)

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

    # ------------------------------------------------------------------
    # Trim helpers
    # ------------------------------------------------------------------

    def hover_wrench(self) -> NDArray[np.floating]:
        """Return the wrench ``[T, 0, 0, 0]`` for hover at current mass."""
        return np.array([self.params.mass * self.params.gravity, 0.0, 0.0, 0.0])

    def hover_forces(self) -> NDArray[np.floating]:
        """Per-rotor thrust [N] that holds a hover.

        Not simply weight over rotor count: the mixer splits the load
        according to the layout, and a coaxial pair whose lower rotor works
        in a wake carries the same force at a higher shaft speed, not a
        smaller share of the load.
        """
        return self.mixer.wrench_to_forces(self.hover_wrench())

    def spin_up_to_hover(self) -> NDArray[np.floating]:
        """Pre-spin every motor to its hover speed, and return those speeds.

        Motors start stopped, so a simulation that begins at a commanded
        hover drops for the first few motor time constants while they come
        up. Call this after :meth:`reset` when the transient is not what is
        being studied.
        """
        forces = self.hover_forces()
        for motor, force in zip(self.motors, forces):
            motor.reset(motor.thrust_to_omega(float(force)))
        return self.get_motor_speeds()

    def get_motor_speeds(self) -> NDArray[np.floating]:
        """Return current motor angular velocities [rad/s]."""
        return np.array([m.omega for m in self.motors])

    def get_rotor_thrusts(self) -> NDArray[np.floating]:
        """Return the thrust each rotor is currently making [N]."""
        return np.array([m.thrust for m in self.motors])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.params.name!r}, "
            f"n_rotors={self.n_rotors}, mass={self.params.mass} kg)"
        )
