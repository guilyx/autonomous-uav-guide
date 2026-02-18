# Erwin Lejeune - 2026-02-17
"""Simplified tilt-rotor VTOL model with transition dynamics.

Reference: R. Bapst et al., "Design and Implementation of an Unmanned
Tail-Sitter," IROS, 2015. Adapted for generic tilt-rotor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from uav_sim.vehicles.base import UAVBase, UAVParams


@dataclass
class TiltrotorParams(UAVParams):
    """Physical parameters for a tilt-rotor VTOL."""

    mass: float = 5.0
    inertia: NDArray[np.floating] = field(default_factory=lambda: np.diag([0.1, 0.1, 0.15]))
    num_rotors: int = 4
    arm_length: float = 0.3
    wing_area: float = 0.4
    CL_alpha: float = 2.0
    CD0: float = 0.05
    rho_air: float = 1.225
    max_tilt: float = np.pi / 2  # 0 = hover, pi/2 = forward


class Tiltrotor(UAVBase):
    """Simplified tilt-rotor VTOL with smooth hover-cruise transition.

    State: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    Control: [total_thrust, tau_x, tau_y, tau_z, tilt_angle]

    The ``tilt_angle`` linearly blends between hover (0) and cruise (pi/2)
    configurations, modifying thrust direction and engaging wing lift.
    """

    def __init__(self, params: TiltrotorParams | None = None) -> None:
        self.vtol_params = params or TiltrotorParams()
        super().__init__(self.vtol_params)

    @property
    def state_dim(self) -> int:
        return 12

    @property
    def control_dim(self) -> int:
        return 5

    def _dynamics(self, state: NDArray[np.floating], control: NDArray[np.floating]):
        p = self.vtol_params
        _, _, _, phi, theta, _, vx, vy, vz, pb, qb, rb = state
        T, tx, ty, tz, tilt = control
        tilt = np.clip(tilt, 0.0, p.max_tilt)

        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, _ = np.cos(theta), np.sin(theta)

        ct, st = np.cos(tilt), np.sin(tilt)
        thrust_body = np.array([T * st, 0.0, T * ct])

        from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

        R = Quadrotor.rotation_matrix(phi, theta, state[5])
        f_thrust = R @ thrust_body

        Va = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-6
        alpha = np.arctan2(-vz, np.sqrt(vx**2 + vy**2) + 1e-12)
        qbar = 0.5 * p.rho_air * Va**2 * p.wing_area
        blend = np.sin(tilt)
        L_wing = qbar * p.CL_alpha * alpha * blend
        D_wing = qbar * p.CD0 * blend

        vel_dir = np.array([vx, vy, vz]) / (Va + 1e-12)
        lift_dir = np.array([0, 0, 1.0]) - vel_dir * vel_dir[2]
        ln = np.linalg.norm(lift_dir)
        if ln > 1e-6:
            lift_dir /= ln
        f_aero = lift_dir * L_wing - vel_dir * D_wing

        acc = (f_thrust + f_aero) / p.mass + np.array([0, 0, -p.gravity])
        dx, dy, dz = vx, vy, vz
        dvx, dvy, dvz = acc

        dphi = pb + (sphi * np.tan(theta)) * qb + (cphi * np.tan(theta)) * rb
        dtheta = cphi * qb - sphi * rb
        dpsi = (sphi / (cth + 1e-12)) * qb + (cphi / (cth + 1e-12)) * rb

        Ix, Iy, Iz = p.inertia[0, 0], p.inertia[1, 1], p.inertia[2, 2]
        dpb = ((Iy - Iz) * qb * rb + tx) / Ix
        dqb = ((Iz - Ix) * pb * rb + ty) / Iy
        drb = ((Ix - Iy) * pb * qb + tz) / Iz

        return np.array([dx, dy, dz, dphi, dtheta, dpsi, dvx, dvy, dvz, dpb, dqb, drb])
