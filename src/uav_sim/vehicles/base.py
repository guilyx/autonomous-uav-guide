# Erwin Lejeune - 2026-02-17
"""Abstract base class for all UAV vehicle models.

Every airframe (multirotor, VTOL, fixed-wing) inherits from :class:`UAVBase`
and implements its own dynamics, actuation, and state conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class UAVParams:
    """Common physical parameters shared by all UAV types."""

    mass: float = 1.0
    gravity: float = 9.81
    inertia: NDArray[np.floating] = field(default_factory=lambda: np.diag([0.01, 0.01, 0.02]))
    drag_coeffs: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))


class UAVBase(ABC):
    """Abstract UAV vehicle model.

    Sub-classes must implement:

    * :meth:`_dynamics` — continuous-time state derivative.
    * :meth:`state_dim` — dimensionality of the state vector.
    * :meth:`control_dim` — dimensionality of the control input.
    """

    def __init__(self, params: UAVParams | None = None) -> None:
        self.params = params or UAVParams()
        self._state: NDArray[np.floating] = np.zeros(self.state_dim)
        self._time: float = 0.0

    # ── abstract interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def state_dim(self) -> int: ...

    @property
    @abstractmethod
    def control_dim(self) -> int: ...

    @abstractmethod
    def _dynamics(
        self,
        state: NDArray[np.floating],
        control: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return dx/dt given current *state* and *control* input."""
        ...

    # ── concrete helpers ──────────────────────────────────────────────────

    @property
    def state(self) -> NDArray[np.floating]:
        return self._state.copy()

    @property
    def position(self) -> NDArray[np.floating]:
        """World-frame position [x, y, z]."""
        return self._state[:3].copy()

    @property
    def time(self) -> float:
        return self._time

    def reset(
        self,
        state: NDArray[np.floating] | None = None,
        time: float = 0.0,
    ) -> None:
        """Reset vehicle to a given state."""
        if state is not None:
            self._state = np.array(state, dtype=np.float64)
        else:
            self._state = np.zeros(self.state_dim)
        self._time = time

    def step(self, control: NDArray[np.floating], dt: float) -> None:
        """Advance the dynamics by *dt* using RK4 integration."""
        s = self._state
        k1 = self._dynamics(s, control)
        k2 = self._dynamics(s + 0.5 * dt * k1, control)
        k3 = self._dynamics(s + 0.5 * dt * k2, control)
        k4 = self._dynamics(s + dt * k3, control)
        self._state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._time += dt
