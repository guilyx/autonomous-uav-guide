# Erwin Lejeune - 2026-02-16
"""Control allocation mixer: body wrench to individual motor forces.

The mixing matrix used to be written out by hand, one literal per frame
type. It is now derived from the rotor layout in
:mod:`uav_sim.vehicles.components.allocation`, which reproduces both
literals to machine precision — see ``tests/test_allocation.py``, which
pins the historical matrices against the derivation. Deriving it is what
lets the same code mix a hexacopter, an octocopter or a coaxial X8.

Reference: R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles,"
IEEE RAM, 2012. DOI: 10.1109/MRA.2012.2206474
"""

from __future__ import annotations

from uav_sim.vehicles.components.allocation import (
    ControlAllocation,
    Rotor,
    plus_layout,
    x_layout,
)


class Mixer(ControlAllocation):
    """Maps ``[T, tau_x, tau_y, tau_z]`` to per-rotor forces and back.

    A thin named entry point onto :class:`ControlAllocation` for the two
    quadrotor frames the library has always shipped. Anything with a rotor
    count other than four, or a layout that is not a symmetric ring, should
    build a :class:`ControlAllocation` from an explicit rotor list instead.

    The X-frame column order is **rear-left, rear-right, front-right,
    front-left**, with spin directions ``CCW, CW, CCW, CW``; the ``+``-frame
    order is rear, right, front, left with the same spins. Both fall out of
    the geometry: they are the orders that reproduce the matrices this
    module used to hard-code.

    Parameters:
        arm_length: Distance from CoM to motor [m].
        k_thrust: Thrust coefficient [N/(rad/s)^2].
        k_torque: Torque coefficient [Nm/(rad/s)^2].
        frame: ``"x"`` for X-frame, ``"+"`` for +-frame.
        rotors: Explicit layout, overriding *frame* and *arm_length*.
        max_thrust: Per-rotor thrust ceiling [N], or ``None`` for unbounded.
        saturation: ``"clip"`` (default) or ``"prioritise_torque"``.
    """

    def __init__(
        self,
        arm_length: float = 0.175,
        k_thrust: float = 8.55e-6,
        k_torque: float = 1.36e-7,
        frame: str = "x",
        rotors: list[Rotor] | None = None,
        max_thrust: float | None = None,
        saturation: str = "clip",
    ) -> None:
        if rotors is None:
            rotors = self.frame_layout(frame, arm_length)
        self.arm_length = arm_length
        self.frame = frame
        super().__init__(
            rotors,
            k_thrust=k_thrust,
            k_torque=k_torque,
            max_thrust=max_thrust,
            saturation=saturation,
        )

    @staticmethod
    def frame_layout(frame: str, arm_length: float, n_rotors: int = 4) -> list[Rotor]:
        """Rotor layout for a named frame type."""
        if frame == "x":
            return x_layout(n_rotors, arm_length)
        if frame == "+":
            return plus_layout(n_rotors, arm_length)
        raise ValueError(f"Unknown frame type: {frame!r}. Use 'x' or '+'.")
