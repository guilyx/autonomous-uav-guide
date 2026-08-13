# Erwin Lejeune - 2026-02-21
"""Linear Model Predictive Controller for quadrotor path tracking.

Uses the same hover-linearised model as the LQR controller, but solves
a finite-horizon optimal control problem at every step, applying only
the first control input (receding horizon).

Includes a ``compute_path_follow`` mode that finds a local nearest-point
on the path and distributes reference points along the prediction horizon,
avoiding the simple "track a single global target" pitfall.

Reference: J. B. Rawlings, D. Q. Mayne, M. M. Diehl, "Model Predictive
Control: Theory, Computation, and Design," 2nd ed., Nob Hill, 2017.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D


class MPCController:
    """Receding-horizon linear MPC for quadrotor position tracking.

    Parameters
    ----------
    horizon : prediction horizon (number of steps).
    dt : discretisation time step [s].
    mass, gravity : quadrotor parameters.
    inertia : 3x3 inertia tensor.
    Q : 12x12 state cost.
    R : 4x4 input cost.
    """

    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.02,
        mass: float = 1.5,
        gravity: float = 9.81,
        inertia: NDArray[np.floating] | None = None,
        Q: NDArray[np.floating] | None = None,
        R: NDArray[np.floating] | None = None,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.mass = mass
        self.gravity = gravity
        inertia = inertia if inertia is not None else np.diag([0.0082, 0.0082, 0.0148])

        A_c, B_c = self._linearise(mass, gravity, inertia)
        # Euler discretisation
        self.A_d = np.eye(12) + A_c * dt
        self.B_d = B_c * dt

        self.Q = Q if Q is not None else np.diag([15, 15, 25, 3, 3, 1, 4, 4, 4, 0.5, 0.5, 0.2])
        self.R = R if R is not None else np.diag([0.05, 0.5, 0.5, 0.5])

        # Terminal cost from the *discrete* ARE. The running cost below is
        # summed per step, not integrated, so a continuous-time Riccati
        # solution under-weights the terminal state by a factor of 1/dt and
        # the horizon effectively ends in mid-air.
        self.Qf = solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)

        self.hover_wrench = np.array([mass * gravity, 0.0, 0.0, 0.0])
        self._warm_start: NDArray[np.floating] | None = None

    def reset(self) -> None:
        self._warm_start = None

    def _reference_horizon(
        self,
        target_pos: NDArray[np.floating],
        target_vel: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """One 12-state reference per horizon step.

        Preview is the whole reason to run MPC instead of LQR. Holding a
        single reference point across the horizon throws it away and asks
        the plan to come to rest where the trajectory happens to be right
        now, which shows up as a fixed lag on any moving reference.
        """
        pos = np.atleast_2d(np.asarray(target_pos, dtype=float))
        if len(pos) == 1:
            pos = np.repeat(pos, self.horizon + 1, axis=0)
        vel = (
            np.zeros_like(pos)
            if target_vel is None
            else np.atleast_2d(np.asarray(target_vel, dtype=float))
        )
        if len(vel) == 1:
            vel = np.repeat(vel, self.horizon + 1, axis=0)

        n = self.horizon + 1
        ref = np.zeros((n, 12))
        ref[: len(pos), :3] = pos[:n]
        ref[: len(vel), 6:9] = vel[:n]
        if len(pos) < n:
            ref[len(pos) :, :3] = pos[-1]
        if len(vel) < n:
            ref[len(vel) :, 6:9] = vel[-1]
        return ref

    def compute(
        self,
        state: NDArray[np.floating],
        target_pos: NDArray[np.floating],
        target_vel: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Compute MPC wrench for one step.

        Parameters
        ----------
        state : 12-element quadrotor state.
        target_pos : desired ``[x, y, z]``, or ``(horizon + 1, 3)`` to give
            the controller preview of a moving reference.
        target_vel : desired ``[vx, vy, vz]``, same shapes (default zeros).

        Returns
        -------
        ``[T, τx, τy, τz]`` wrench.
        """
        ref_traj = self._reference_horizon(target_pos, target_vel)

        # The hover linearisation is affine in the absolute state — A only
        # couples position to velocity and attitude to rates, and gravity is
        # already carried by the hover thrust — so the horizon is rolled out
        # in absolute coordinates and each step scored against its own
        # reference.
        x0 = np.asarray(state, dtype=float).copy()
        H = self.horizon
        u_dim = 4

        if self._warm_start is not None and len(self._warm_start) == H * u_dim:
            u0 = self._warm_start.copy()
        else:
            u0 = np.zeros(H * u_dim)

        bounds = []
        for _ in range(H):
            bounds.append((0.0 - self.hover_wrench[0], self.mass * self.gravity * 1.5))
            for _ in range(3):
                bounds.append((-0.5, 0.5))

        result = minimize(
            self._cost,
            u0,
            args=(x0, ref_traj),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 15, "ftol": 1e-5},
        )
        u_opt = result.x.reshape(H, u_dim)

        # Warm-start: shift solution
        self._warm_start = np.zeros(H * u_dim)
        self._warm_start[: (H - 1) * u_dim] = result.x[u_dim:]

        return self.hover_wrench + u_opt[0]

    # ------------------------------------------------------------------
    # Path-following mode
    # ------------------------------------------------------------------

    def compute_path_follow(
        self,
        state: NDArray[np.floating],
        path: NDArray[np.floating],
        path_index: int = 0,
        lookahead: float = 3.0,
        speed: float = 1.5,
    ) -> tuple[NDArray[np.floating], int]:
        """Track a local segment of *path* rather than a single point.

        Finds the nearest segment via sphere-segment intersection, then
        delegates to :meth:`compute` with an interpolated target + velocity.

        Returns ``(wrench, updated_index)`` so the caller can persist the
        segment index across calls.
        """
        pos = state[:3]
        n = len(path)
        if n == 0:
            return self.compute(state, pos), path_index

        while path_index < n - 1:
            if np.linalg.norm(pos - path[path_index]) < lookahead * 0.5:
                path_index += 1
            else:
                break

        target: NDArray[np.floating] | None = None
        for i in range(path_index, n - 1):
            pt = PurePursuit3D._intersect_sphere_segment(pos, lookahead, path[i], path[i + 1])
            if pt is not None:
                target = pt
                break

        if target is None:
            target = path[min(path_index, n - 1)].copy()

        direction = target - pos
        dist = float(np.linalg.norm(direction))
        vel = direction / max(dist, 1e-6) * speed if dist > 0.01 else np.zeros(3)

        wrench = self.compute(state, target, target_vel=vel)
        return wrench, path_index

    def _cost(
        self,
        u_flat: NDArray[np.floating],
        x0: NDArray[np.floating],
        ref_traj: NDArray[np.floating],
    ) -> float:
        H = self.horizon
        u = u_flat.reshape(H, 4)
        x = x0.copy()
        cost = 0.0
        for k in range(H):
            e = x - ref_traj[k]
            cost += float(e @ self.Q @ e + u[k] @ self.R @ u[k])
            x = self.A_d @ x + self.B_d @ u[k]
        e = x - ref_traj[H]
        cost += float(e @ self.Qf @ e)
        return cost

    @staticmethod
    def _linearise(
        mass: float, gravity: float, inertia: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        Ix, Iy, Iz = np.diag(inertia)
        A = np.zeros((12, 12))
        A[0, 6] = 1.0
        A[1, 7] = 1.0
        A[2, 8] = 1.0
        A[6, 4] = gravity
        A[7, 3] = -gravity
        A[3, 9] = 1.0
        A[4, 10] = 1.0
        A[5, 11] = 1.0

        B = np.zeros((12, 4))
        B[8, 0] = 1.0 / mass
        B[9, 1] = 1.0 / Ix
        B[10, 2] = 1.0 / Iy
        B[11, 3] = 1.0 / Iz
        return A, B
