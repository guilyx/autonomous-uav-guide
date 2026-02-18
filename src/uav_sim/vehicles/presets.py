# Erwin Lejeune - 2026-02-15
"""Ready-made vehicle configurations for common quadrotor platforms.

Each preset provides physically-validated :class:`QuadrotorParams` so
users can spin up a realistic simulation with a single call::

    quad = create_quadrotor(VehiclePreset.CRAZYFLIE)
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from uav_sim.vehicles.multirotor.quadrotor import Quadrotor, QuadrotorParams


class VehiclePreset(Enum):
    """Catalogue of supported drone platforms."""

    CRAZYFLIE = "crazyflie"
    DJI_MINI = "dji_mini"
    RACING_250 = "racing_250"
    DJI_MATRICE = "dji_matrice"
    CUSTOM = "custom"


_PRESETS: dict[VehiclePreset, QuadrotorParams] = {
    VehiclePreset.CRAZYFLIE: QuadrotorParams(
        mass=0.027,
        arm_length=0.046,
        inertia=np.diag([1.66e-5, 1.66e-5, 2.96e-5]),
        k_thrust=2.55e-8,
        k_torque=7.94e-10,
        motor_tau=0.02,
        omega_max=2500.0,
        drag_coeff=0.01,
    ),
    VehiclePreset.DJI_MINI: QuadrotorParams(
        mass=0.249,
        arm_length=0.11,
        inertia=np.diag([6.5e-4, 6.5e-4, 1.2e-3]),
        k_thrust=1.0e-7,
        k_torque=3.2e-9,
        motor_tau=0.025,
        omega_max=1800.0,
        drag_coeff=0.03,
    ),
    VehiclePreset.RACING_250: QuadrotorParams(
        mass=1.5,
        arm_length=0.175,
        inertia=np.diag([0.0082, 0.0082, 0.0148]),
        k_thrust=8.55e-6,
        k_torque=1.36e-7,
        motor_tau=0.02,
        omega_max=1100.0,
        drag_coeff=0.1,
    ),
    VehiclePreset.DJI_MATRICE: QuadrotorParams(
        mass=3.6,
        arm_length=0.32,
        inertia=np.diag([0.045, 0.045, 0.080]),
        k_thrust=3.0e-5,
        k_torque=5.0e-7,
        motor_tau=0.03,
        omega_max=800.0,
        drag_coeff=0.15,
    ),
}


def create_quadrotor(
    preset: VehiclePreset = VehiclePreset.RACING_250,
    **overrides: float,
) -> Quadrotor:
    """Create a :class:`Quadrotor` from a named preset.

    Any keyword argument overrides the corresponding field in
    :class:`QuadrotorParams` (e.g. ``mass=2.0``).

    Parameters
    ----------
    preset : which platform to use.
    **overrides : per-field overrides forwarded to ``QuadrotorParams``.

    Returns
    -------
    Quadrotor
        Ready-to-fly quadrotor instance.
    """
    if preset == VehiclePreset.CUSTOM:
        params = QuadrotorParams(**overrides)
    else:
        base = _PRESETS[preset]
        if overrides:
            kw = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
            kw.update(overrides)
            params = QuadrotorParams(**kw)
        else:
            params = base
    return Quadrotor(params)


def get_params(preset: VehiclePreset) -> QuadrotorParams:
    """Return the :class:`QuadrotorParams` for a named preset."""
    return _PRESETS[preset]
