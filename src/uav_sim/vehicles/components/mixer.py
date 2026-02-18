# Erwin Lejeune - 2026-02-16
"""Control allocation mixer: body wrench to individual motor forces.

Reference: R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles,"
IEEE RAM, 2012. DOI: 10.1109/MRA.2012.2206474
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Mixer:
    """Maps ``[T, tau_x, tau_y, tau_z]`` to ``[f1, f2, f3, f4]`` and back.

    Supports X-frame and +-frame quadrotor configurations.

    Parameters:
        arm_length: Distance from CoM to motor [m].
        k_thrust: Thrust coefficient [N/(rad/s)^2].
        k_torque: Torque coefficient [Nm/(rad/s)^2].
        frame: ``"x"`` for X-frame, ``"+"`` for +-frame.
    """

    def __init__(
        self,
        arm_length: float = 0.0397,
        k_thrust: float = 2.98e-6,
        k_torque: float = 1.14e-7,
        frame: str = "x",
    ) -> None:
        self.arm_length = arm_length
        self.k_thrust = k_thrust
        self.k_torque = k_torque
        self.frame = frame
        self._mix_matrix = self._build_mix_matrix()
        self._inv_mix_matrix = np.linalg.inv(self._mix_matrix)

    def _build_mix_matrix(self) -> NDArray[np.floating]:
        """Build the 4x4 mixing matrix (wrench → forces)."""
        L = self.arm_length
        kappa = self.k_torque / self.k_thrust  # torque-to-thrust ratio

        if self.frame == "x":
            s = L / np.sqrt(2.0)
            # Motor order: FL(CCW), FR(CW), RR(CCW), RL(CW)
            return np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [s, -s, -s, s],
                    [s, s, -s, -s],
                    [-kappa, kappa, -kappa, kappa],
                ]
            )
        elif self.frame == "+":
            return np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, -L, 0.0, L],
                    [L, 0.0, -L, 0.0],
                    [-kappa, kappa, -kappa, kappa],
                ]
            )
        else:
            raise ValueError(f"Unknown frame type: {self.frame!r}. Use 'x' or '+'.")

    def wrench_to_forces(self, wrench: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert body wrench ``[T, τx, τy, τz]`` to motor forces ``[f1..f4]``.

        Forces are clamped to non-negative values.
        """
        forces = self._inv_mix_matrix @ np.asarray(wrench, dtype=np.float64)
        return np.maximum(forces, 0.0)

    def forces_to_wrench(self, forces: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert motor forces ``[f1..f4]`` to body wrench ``[T, τx, τy, τz]``."""
        return self._mix_matrix @ np.asarray(forces, dtype=np.float64)

    @property
    def mix_matrix(self) -> NDArray[np.floating]:
        """The 4x4 mixing matrix."""
        return self._mix_matrix.copy()

    @property
    def inv_mix_matrix(self) -> NDArray[np.floating]:
        """Inverse mixing matrix (wrench → forces)."""
        return self._inv_mix_matrix.copy()
