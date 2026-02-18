# Erwin Lejeune - 2026-02-17
"""Simplified fixed-wing aerodynamic model.

Reference: R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and
Practice," Princeton University Press, 2012, Chapters 3-4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from uav_sim.vehicles.base import UAVBase, UAVParams


@dataclass
class FixedWingParams(UAVParams):
    """Physical parameters for a fixed-wing aircraft."""

    wing_area: float = 0.55
    wing_span: float = 2.9
    chord: float = 0.19
    e_oswald: float = 0.9
    CL0: float = 0.28
    CLa: float = 3.45
    CD0: float = 0.03
    CDa: float = 0.30
    Cm0: float = -0.02
    Cma: float = -0.38
    rho_air: float = 1.225
    mass: float = 13.5
    inertia: NDArray[np.floating] = field(default_factory=lambda: np.diag([0.8244, 1.135, 1.759]))


class FixedWing(UAVBase):
    """Simplified 6DOF fixed-wing aircraft.

    State: [x, y, z, phi, theta, psi, u, v, w, p, q, r]
    Control: [delta_e, delta_a, delta_r, delta_t] (elevator, aileron, rudder, throttle)
    """

    def __init__(self, params: FixedWingParams | None = None) -> None:
        self.fw_params = params or FixedWingParams()
        super().__init__(self.fw_params)

    @property
    def state_dim(self) -> int:
        return 12

    @property
    def control_dim(self) -> int:
        return 4

    def _dynamics(self, state: NDArray[np.floating], control: NDArray[np.floating]):
        p = self.fw_params
        x, y, z, phi, theta, psi, u, v, w, pb, qb, rb = state
        Va = np.sqrt(u**2 + v**2 + w**2) + 1e-6
        alpha = np.arctan2(w, u + 1e-12)
        qbar = 0.5 * p.rho_air * Va**2 * p.wing_area

        CL = p.CL0 + p.CLa * alpha
        CD = p.CD0 + p.CDa * alpha**2
        L_aero = qbar * CL
        D_aero = qbar * CD

        fx = -D_aero * np.cos(alpha) + L_aero * np.sin(alpha) + control[3] * p.mass * p.gravity
        fz = -D_aero * np.sin(alpha) - L_aero * np.cos(alpha)

        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        du = rb * v - qb * w + fx / p.mass - p.gravity * sth
        dv = pb * w - rb * u + p.gravity * cth * sphi
        dw = qb * u - pb * v + fz / p.mass + p.gravity * cth * cphi

        dx = (
            (cth * cpsi) * u
            + (sphi * sth * cpsi - cphi * spsi) * v
            + (cphi * sth * cpsi + sphi * spsi) * w
        )
        dy = (
            (cth * spsi) * u
            + (sphi * sth * spsi + cphi * cpsi) * v
            + (cphi * sth * spsi - sphi * cpsi) * w
        )
        dz = -sth * u + sphi * cth * v + cphi * cth * w

        dphi = pb + (sphi * np.tan(theta)) * qb + (cphi * np.tan(theta)) * rb
        dtheta = cphi * qb - sphi * rb
        dpsi = (sphi / (cth + 1e-12)) * qb + (cphi / (cth + 1e-12)) * rb

        Ix, Iy, Iz = p.inertia[0, 0], p.inertia[1, 1], p.inertia[2, 2]
        dpb = ((Iy - Iz) * qb * rb) / Ix + qbar * p.wing_span * control[1] * 0.1 / Ix
        dqb = ((Iz - Ix) * pb * rb) / Iy + qbar * p.chord * control[0] * 0.1 / Iy
        drb = ((Ix - Iy) * pb * qb) / Iz + qbar * p.wing_span * control[2] * 0.1 / Iz

        return np.array([dx, dy, dz, dphi, dtheta, dpsi, du, dv, dw, dpb, dqb, drb])
