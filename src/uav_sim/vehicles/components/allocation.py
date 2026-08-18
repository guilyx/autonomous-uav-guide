# Erwin Lejeune - 2026-08-18
"""Control allocation for an arbitrary rotor layout.

A multirotor is fully described, for allocation purposes, by where its
rotors sit and which way they spin. Everything else — the 4xN allocation
matrix, its pseudo-inverse, how much yaw authority the airframe has, which
wrench axes it cannot reach at all — follows from those two facts. This
module derives them rather than tabulating a matrix per airframe.

Rotor ``i`` sits at body position ``r_i`` (FLU) and pushes along body
``+z`` with force ``f_i >= 0``. Its wrench contribution is

.. math::

    F_i = f_i \\hat{z}, \\qquad
    \\tau_i = r_i \\times F_i - \\sigma_i \\kappa f_i \\hat{z}

with ``sigma_i = +1`` for a counter-clockwise rotor seen from above and
``kappa = k_torque / k_thrust`` the rotor's torque-to-thrust ratio. Writing
``r_i = (x_i, y_i, z_i)`` and expanding the cross product gives the column
of the allocation matrix::

    [ T  ]   [   1    ]
    [ tx ] = [   y_i  ] f_i
    [ ty ]   [  -x_i  ]
    [ tz ]   [ -k s_i ]

Note that ``z_i`` drops out: a rotor thrusting along ``+z`` produces no
moment about ``z`` from its lever arm, only the reaction torque of its own
drag. That is precisely why stacking two rotors coaxially adds thrust and
yaw authority but no roll or pitch authority, and why a layout whose
rotors are collinear in the ``xy`` plane is rank-deficient no matter how
many of them there are.

The signs are not free. They follow from the library's **FLU** body frame
(``x`` forward, ``y`` left, ``z`` up), in which positive pitch is nose-down
— see :doc:`/guide/conventions`. A rotor at the front (``x_i > 0``)
produces ``tau_y = -x_i f_i < 0``, which is the nose going *up*. Ported
from a Forward-Right-Down text, every one of these rows would need a sign
flip.

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
- J. G. Leishman, "Principles of Helicopter Aerodynamics," 2nd ed.,
  Cambridge University Press, 2006, Sec. 2.14 (coaxial rotor interference).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Rotor",
    "ControlAllocation",
    "allocation_matrix",
    "radial_layout",
    "x_layout",
    "plus_layout",
    "h_layout",
    "coaxial_layout",
    "WRENCH_AXES",
]

WRENCH_AXES = ("thrust", "roll", "pitch", "yaw")
"""Row order of the allocation matrix, matching the ``[T, tx, ty, tz]`` wrench."""

#: Below this fraction of the largest singular value an axis counts as
#: unreachable. Loose enough that a legitimately weak yaw channel (kappa is
#: three orders of magnitude below the arm length) still reads as actuated,
#: tight enough to catch a genuinely collapsed one.
_RANK_RTOL = 1e-9


@dataclass(eq=False)
class Rotor:
    """One rotor: where it is, which way it turns, how well it works.

    Parameters:
        position: ``[x, y, z]`` of the rotor hub in body FLU coordinates [m].
        direction: ``+1`` for counter-clockwise seen from above (angular
            velocity along body ``+z``), ``-1`` for clockwise. This sets the
            sign of the reaction torque the airframe feels, and therefore
            the sign of the rotor's yaw authority.
        thrust_scale: Multiplier on both the thrust and the torque this
            rotor develops for a given shaft speed, ``1.0`` for a rotor in
            clean air. The lower rotor of a coaxial pair works in the wake
            of the upper one and typically returns 0.80-0.90; see Leishman
            Sec. 2.14. Scaling thrust and torque together is a first-order
            lump, not a wake model.
        label: Optional human-readable name, used in diagnostics only.
    """

    position: NDArray[np.floating]
    direction: int = 1
    thrust_scale: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        # Copied and frozen: preset layouts are shared between aircraft, and
        # a rotor that moves under one of them would silently move under all
        # of them — and leave the allocation matrix describing neither.
        self.position = np.array(self.position, dtype=np.float64).reshape(3)
        self.position.setflags(write=False)
        if self.direction not in (1, -1):
            raise ValueError(f"Rotor direction must be +1 (CCW) or -1 (CW), got {self.direction!r}")
        if self.thrust_scale <= 0.0:
            raise ValueError(f"Rotor thrust_scale must be positive, got {self.thrust_scale!r}")

    @property
    def x(self) -> float:
        """Forward offset from the centre of mass [m]."""
        return float(self.position[0])

    @property
    def y(self) -> float:
        """Left offset from the centre of mass [m]."""
        return float(self.position[1])

    @property
    def z(self) -> float:
        """Vertical offset from the centre of mass [m]. Does not affect allocation."""
        return float(self.position[2])

    @property
    def arm_length(self) -> float:
        """Distance from the centre of mass to the rotor axis [m]."""
        return float(np.hypot(self.position[0], self.position[1]))


# ----------------------------------------------------------------------
# Layout builders
# ----------------------------------------------------------------------


def radial_layout(
    n_rotors: int,
    arm_length: float,
    offset: float = 0.0,
    first_direction: int = 1,
) -> list[Rotor]:
    """Rotors evenly spaced on a circle, spinning alternately.

    Args:
        n_rotors: Number of rotors. Must be even and at least 4 — an odd
            ring cannot alternate spin directions all the way round, so it
            would carry a residual yaw bias at hover.
        arm_length: Radius from the centre of mass to each rotor [m].
        offset: Angle of the first rotor measured from body ``+x``
            (forward), increasing counter-clockwise toward ``+y`` (left)
            [rad].
        first_direction: Spin of the first rotor, ``+1`` (CCW) or ``-1``.

    Returns:
        Rotors in counter-clockwise order starting at *offset*.
    """
    if n_rotors < 4 or n_rotors % 2 != 0:
        raise ValueError(
            f"A radial layout needs an even number of at least 4 rotors, got {n_rotors}. "
            "An odd ring cannot alternate spin direction, leaving a net yaw torque at hover."
        )
    if arm_length <= 0.0:
        raise ValueError(f"arm_length must be positive, got {arm_length}")

    rotors = []
    for k in range(n_rotors):
        angle = offset + k * 2.0 * np.pi / n_rotors
        position = np.array([arm_length * np.cos(angle), arm_length * np.sin(angle), 0.0])
        # sin(pi) is 1.2e-16, not 0, so a rotor placed dead astern lands a
        # fraction of a femtometre off the centreline and shows up in the
        # roll row of the allocation matrix. Snapping is a better
        # description of the airframe than the round-off is.
        position[np.abs(position) < 1e-12 * arm_length] = 0.0
        rotors.append(
            Rotor(
                position=position,
                direction=first_direction * (-1) ** k,
                label=f"r{k}",
            )
        )
    return rotors


def plus_layout(n_rotors: int = 4, arm_length: float = 0.175) -> list[Rotor]:
    """``+`` frame: one rotor on the ``-x`` (rear) axis, the rest evenly round.

    The first rotor sits directly aft so that, for ``n_rotors == 4``, the
    order is rear, right, front, left — which is the column order of the
    ``+``-frame mixer this library has always shipped.
    """
    return radial_layout(n_rotors, arm_length, offset=np.pi)


def x_layout(n_rotors: int = 4, arm_length: float = 0.175) -> list[Rotor]:
    """``X`` frame: the ``+`` frame rotated half a sector, so nothing sits on the nose.

    For ``n_rotors == 4`` the order is rear-left, rear-right, front-right,
    front-left with spins ``+ - + -``; for 6 it is the usual flat "hexa X"
    with rotors at 30 degree offsets from the lateral axis.
    """
    return radial_layout(n_rotors, arm_length, offset=np.pi - np.pi / n_rotors)


def h_layout(
    n_rotors: int = 4,
    length: float = 0.248,
    width: float = 0.248,
    first_direction: int = 1,
) -> list[Rotor]:
    """``H`` frame: rotors on two rails, longer fore-aft than side-to-side.

    An H frame trades roll authority for a wider fuselage bay. Rotors are
    laid out as ``n_rotors / 2`` rows evenly spaced from the nose to the
    tail, each row carrying a left and a right rotor.

    Args:
        n_rotors: Even, at least 4.
        length: Total fore-aft separation between the front and rear rows [m].
        width: Total lateral separation between the left and right rails [m].
        first_direction: Spin of the front-left rotor.

    Returns:
        Rotors ordered rear to front, left rail before right within a row.
    """
    if n_rotors < 4 or n_rotors % 2 != 0:
        raise ValueError(f"An H layout needs an even number of at least 4 rotors, got {n_rotors}")
    if length <= 0.0 or width <= 0.0:
        raise ValueError(f"H layout length and width must be positive, got {length}, {width}")

    rows = n_rotors // 2
    xs = np.linspace(-length / 2.0, length / 2.0, rows)
    half_width = width / 2.0

    rotors = []
    for k, x in enumerate(xs):
        # Alternate the spin down each rail so the pattern sums to zero and
        # diagonally opposite rotors co-rotate, exactly as on an X frame.
        left = first_direction * (-1) ** k
        rotors.append(Rotor(np.array([x, half_width, 0.0]), left, label=f"L{k}"))
        rotors.append(Rotor(np.array([x, -half_width, 0.0]), -left, label=f"R{k}"))
    return rotors


def coaxial_layout(
    base: list[Rotor],
    separation: float = 0.10,
    lower_efficiency: float = 0.85,
) -> list[Rotor]:
    """Stack a second, counter-rotating rotor under each rotor of *base*.

    This is how an X8 is built: four arms, eight rotors. The pair shares an
    ``(x, y)`` position, so it doubles the available thrust without adding
    any roll or pitch authority — the extra columns of the allocation
    matrix lie in directions the first four already span. What it does add
    is redundancy: the pseudo-inverse spreads a commanded wrench over both
    members of every pair, so losing one costs half an arm rather than a
    whole one.

    Args:
        base: The upper rotors.
        separation: Vertical distance between the two discs [m]. Recorded on
            the rotor for drawing and inertia bookkeeping; it does not enter
            the allocation, because a rotor pushing along ``+z`` has no
            moment arm about ``z``.
        lower_efficiency: Thrust and torque the lower rotor returns relative
            to the upper one, working in its wake. 0.80-0.90 is typical.

    Returns:
        Upper and lower rotor of each arm, adjacent in the list.
    """
    if separation < 0.0:
        raise ValueError(f"Coaxial separation must be non-negative, got {separation}")

    rotors = []
    for rotor in base:
        upper = rotor.position + np.array([0.0, 0.0, separation / 2.0])
        lower = rotor.position - np.array([0.0, 0.0, separation / 2.0])
        name = rotor.label or f"a{len(rotors) // 2}"
        rotors.append(Rotor(upper, rotor.direction, rotor.thrust_scale, f"{name}-top"))
        rotors.append(
            Rotor(lower, -rotor.direction, rotor.thrust_scale * lower_efficiency, f"{name}-bot")
        )
    return rotors


# ----------------------------------------------------------------------
# The allocation matrix
# ----------------------------------------------------------------------


def allocation_matrix(rotors: list[Rotor], kappa: float) -> NDArray[np.floating]:
    """Build the ``4 x n`` matrix mapping rotor thrusts to a body wrench.

    Args:
        rotors: Rotor layout in body FLU coordinates.
        kappa: Torque-to-thrust ratio ``k_torque / k_thrust`` [m].

    Returns:
        ``A`` such that ``[T, tau_x, tau_y, tau_z] = A @ forces``.
    """
    if not rotors:
        raise ValueError("A multirotor needs at least one rotor")
    if kappa < 0.0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")

    return np.array(
        [
            [1.0 for _ in rotors],
            [r.y for r in rotors],
            [-r.x for r in rotors],
            [-kappa * r.direction for r in rotors],
        ]
    )


@dataclass(frozen=True, eq=False)
class SaturationReport:
    """What :meth:`ControlAllocation.wrench_to_forces` had to give up."""

    requested: NDArray[np.floating]
    achieved: NDArray[np.floating]
    collective_shift: float = 0.0
    clipped: bool = False

    @property
    def error(self) -> NDArray[np.floating]:
        """``achieved - requested``, per wrench axis."""
        return self.achieved - self.requested


class ControlAllocation:
    """Derived mixer for an arbitrary rotor layout.

    Forward, ``forces_to_wrench``, is exact: it is the physics of the
    layout. The inverse is not, and cannot be, for two separate reasons.

    **Redundancy.** With more than four rotors the allocation matrix has a
    non-trivial null space — a whole family of thrust vectors produces the
    same wrench. This class resolves it with the Moore-Penrose
    pseudo-inverse, which picks the minimum-norm member of that family, and
    so the least total rotor effort (Johansen & Fossen, 2013, Sec. 3.1).

    **Saturation.** A rotor cannot pull, and cannot exceed its maximum
    thrust. The unconstrained solution routinely asks it to. Two strategies:

    ``"clip"``
        Clamp each rotor into its feasible range and accept whatever wrench
        that produces. Cheap, and what this library has always done, so it
        stays the default. It is also the worst of the two: clamping one
        rotor perturbs *every* axis of the delivered wrench, including
        attitude, at exactly the moment attitude matters most.

    ``"prioritise_torque"``
        Shift the whole thrust vector by a constant before clamping. On a
        layout whose rotor positions and spin directions sum to zero — every
        preset here, and every sensible airframe — a uniform shift changes
        total thrust and *nothing else*, because the roll, pitch and yaw
        rows of the allocation matrix each sum to zero. So the three torques
        survive intact and the collective absorbs the whole error. This is
        the "air mode" of the open-source flight stacks, and the right
        trade: a quadrotor that loses a little altitude authority recovers;
        one that loses roll authority does not.

    Parameters:
        rotors: Layout in body FLU coordinates.
        k_thrust: Thrust coefficient ``T = k_thrust * omega^2`` [N/(rad/s)^2].
        k_torque: Reaction torque coefficient [Nm/(rad/s)^2].
        max_thrust: Per-rotor thrust ceiling [N], or ``None`` for unbounded.
            A scalar applies to every rotor before its ``thrust_scale``.
        saturation: ``"clip"`` or ``"prioritise_torque"``.
    """

    def __init__(
        self,
        rotors: list[Rotor],
        k_thrust: float = 8.55e-6,
        k_torque: float = 1.36e-7,
        max_thrust: float | None = None,
        saturation: str = "clip",
    ) -> None:
        if saturation not in ("clip", "prioritise_torque"):
            raise ValueError(
                f"Unknown saturation strategy {saturation!r}. "
                "Use 'clip' or 'prioritise_torque'."
            )
        if k_thrust <= 0.0:
            raise ValueError(f"k_thrust must be positive, got {k_thrust}")

        self.rotors = list(rotors)
        self.k_thrust = k_thrust
        self.k_torque = k_torque
        self.kappa = k_torque / k_thrust
        self.saturation = saturation
        self.max_thrust = max_thrust

        self._allocation = allocation_matrix(self.rotors, self.kappa)

        singular = np.linalg.svd(self._allocation, compute_uv=False)
        self._rank = int(np.sum(singular > singular[0] * _RANK_RTOL))
        self._singular_values = singular

        # Four rotors and full rank is the square case, where the
        # pseudo-inverse *is* the inverse. Taking that route explicitly
        # keeps a quadrotor bit-for-bit on the trajectory it flew before
        # the mixer was derived, rather than a few ulp beside it.
        if self._allocation.shape[0] == self._allocation.shape[1] and self._rank == 4:
            self._pinv = np.linalg.inv(self._allocation)
        else:
            self._pinv = np.linalg.pinv(self._allocation)

        # Per-rotor ceiling, after the coaxial efficiency of each rotor.
        if max_thrust is None:
            self._upper: NDArray[np.floating] | None = None
        else:
            if max_thrust <= 0.0:
                raise ValueError(f"max_thrust must be positive, got {max_thrust}")
            self._upper = np.array([max_thrust * r.thrust_scale for r in self.rotors])

        self.last_saturation: SaturationReport | None = None

    # ── geometry ──────────────────────────────────────────────────────

    @property
    def n_rotors(self) -> int:
        return len(self.rotors)

    @property
    def allocation(self) -> NDArray[np.floating]:
        """The ``4 x n`` allocation matrix (forces to wrench)."""
        return self._allocation.copy()

    @property
    def mix_matrix(self) -> NDArray[np.floating]:
        """Alias of :attr:`allocation`, for the historical ``Mixer`` interface."""
        return self._allocation.copy()

    @property
    def inv_mix_matrix(self) -> NDArray[np.floating]:
        """The ``n x 4`` pseudo-inverse (wrench to forces).

        For a four-rotor layout the allocation matrix is square and
        invertible, and this is its exact inverse to machine precision.
        """
        return self._pinv.copy()

    @property
    def rank(self) -> int:
        """Number of wrench axes the layout can reach, out of four."""
        return self._rank

    @property
    def singular_values(self) -> NDArray[np.floating]:
        """Singular values of the allocation matrix, largest first."""
        return self._singular_values.copy()

    @property
    def fully_actuated(self) -> bool:
        """True when thrust and all three torques are independently reachable."""
        return self._rank == 4

    @property
    def unreachable_axes(self) -> tuple[str, ...]:
        """Wrench axes on which a *pure* unit wrench cannot be produced.

        An axis is listed when no thrust vector delivers that axis alone
        with all three others at zero — formally, when ``A A+ e`` differs
        from ``e``. Whatever the airframe is short of, this names it: a
        layout whose rotors are collinear loses ``"roll"`` or ``"pitch"``,
        and one whose rotors all spin the same way loses both ``"yaw"``
        *and* ``"thrust"``, since every newton it lifts with drags a fixed
        reaction torque along with it.
        """
        projector = self._allocation @ self._pinv
        residual = np.eye(4) - projector
        return tuple(
            axis for axis, row in zip(WRENCH_AXES, residual) if np.linalg.norm(row) > 1e-8
        )

    @property
    def yaw_authority(self) -> float:
        """Yaw torque per newton of thrust redistributed between rotors [Nm/N].

        The yaw row of the allocation matrix with its thrust component
        projected out, normalised by the rotor count: how much yaw torque
        the airframe gets out of shuffling thrust around at constant total
        thrust. It equals ``kappa`` for any layout with a balanced
        alternating spin pattern, and falls to zero when every rotor spins
        the same way — the yaw row is then a multiple of the thrust row and
        the two axes stop being independent. This is a property of the spin
        *pattern*, not of the arm length: yaw comes from rotor drag, which
        has no lever arm.
        """
        yaw_row = self._allocation[3]
        thrust_row = self._allocation[0]
        residual = yaw_row - thrust_row * (yaw_row @ thrust_row) / (thrust_row @ thrust_row)
        return float(np.linalg.norm(residual) / np.sqrt(self.n_rotors))

    # ── allocation ────────────────────────────────────────────────────

    def forces_to_wrench(self, forces: NDArray[np.floating]) -> NDArray[np.floating]:
        """Body wrench ``[T, tau_x, tau_y, tau_z]`` produced by rotor forces [N]."""
        forces = np.asarray(forces, dtype=np.float64)
        if forces.shape != (self.n_rotors,):
            raise ValueError(f"Expected {self.n_rotors} rotor forces, got {forces.shape}")
        return self._allocation @ forces

    def wrench_to_forces(self, wrench: NDArray[np.floating]) -> NDArray[np.floating]:
        """Rotor forces [N] that best produce the requested body wrench.

        The result is always feasible: non-negative, and below
        :attr:`max_thrust` when one was given. Which part of the request is
        sacrificed to get there depends on the saturation strategy; the
        cost is recorded in :attr:`last_saturation`.
        """
        wrench = np.asarray(wrench, dtype=np.float64).reshape(4)
        forces = self._pinv @ wrench

        shift = 0.0
        if self.saturation == "prioritise_torque":
            forces, shift = self._shift_into_range(forces)

        feasible = np.maximum(forces, 0.0)
        if self._upper is not None:
            feasible = np.minimum(feasible, self._upper)

        clipped = bool(np.any(np.abs(feasible - forces) > 1e-12))
        self.last_saturation = SaturationReport(
            requested=wrench,
            achieved=self._allocation @ feasible,
            collective_shift=shift,
            clipped=clipped,
        )
        return feasible

    def _shift_into_range(
        self, forces: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float]:
        """Slide every rotor by one constant to fit the feasible band.

        Adding ``d`` to every rotor adds ``n * d`` to the total thrust and
        leaves the torques untouched whenever the roll, pitch and yaw rows
        each sum to zero — the condition :attr:`torque_rows_balanced`
        checks. Any shift in ``[lowest, highest]`` is admissible; the one
        closest to zero is chosen, so a request that already fits is left
        exactly where the pseudo-inverse put it. Where the spread is wider
        than the band itself no shift saves it, and the clamp that follows
        does the rest.
        """
        lowest = -float(np.min(forces))
        highest = np.inf if self._upper is None else float(np.min(self._upper - forces))

        if lowest > highest:
            return forces, 0.0

        shift = float(np.clip(0.0, lowest, highest))
        if shift == 0.0:
            return forces, 0.0
        return forces + shift, shift

    @property
    def torque_rows_balanced(self) -> bool:
        """True when a uniform thrust shift is a pure collective change.

        Requires the roll, pitch and yaw rows to sum to zero, i.e. the
        rotors are balanced about the centre of mass and their spin
        directions cancel. ``"prioritise_torque"`` only keeps its promise on
        such a layout.
        """
        return bool(np.all(np.abs(self._allocation[1:].sum(axis=1)) < 1e-9))

    # ── diagnostics ───────────────────────────────────────────────────

    def hover_forces(self, weight: float) -> NDArray[np.floating]:
        """Rotor forces [N] holding *weight* newtons with no torque."""
        return self.wrench_to_forces(np.array([weight, 0.0, 0.0, 0.0]))

    def __repr__(self) -> str:
        spins = "".join("+" if r.direction > 0 else "-" for r in self.rotors)
        return (
            f"{type(self).__name__}(n_rotors={self.n_rotors}, rank={self.rank}, "
            f"spins='{spins}', saturation={self.saturation!r})"
        )
