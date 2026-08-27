# Erwin Lejeune - 2026-02-15
"""Nonlinear Model Predictive Controller for quadrotor trajectory tracking.

Unlike the linear MPC, this controller predicts with the **full nonlinear**
translational and attitude kinematics — no linearisation about hover — via
direct single-shooting and ``scipy.optimize.minimize``.

Three design choices carry the whole controller:

**Thrust and body rates are the decision variables, not torques.** This is
the standard quadrotor NMPC formulation (Falanga et al., PAMPC 2018) and it
exists because of a timescale problem. Torque moves attitude in tens of
milliseconds and position in seconds; a horizon short enough to integrate
torque stably cannot see position at all, and one long enough to see
position integrates the rotational dynamics straight to infinity. Taking
body rates as inputs removes the stiff dynamics from the prediction model
entirely and delegates them to a rate loop that runs underneath — exactly
how a real flight stack is layered. The controller still returns a wrench,
so callers see no difference.

**The horizon reaches the dynamics being controlled.** Prediction step is
decoupled from control step: predict coarsely over ~1 s, re-plan at 20 Hz.

**Move blocking keeps that horizon affordable.** ``L-BFGS-B`` builds its
gradient by finite differences, so cost scales with the number of decision
variables. Holding the input constant over blocks of prediction steps buys
horizon length nearly for free.

Reference: M. Diehl et al., "Real-Time Optimization and Nonlinear Model
Predictive Control of Processes Governed by Differential-Algebraic
Equations," J. Process Control, 2002. DOI: 10.1016/S0959-1524(02)00023-1

Reference: D. Falanga et al., "PAMPC: Perception-Aware Model Predictive
Control for Quadrotors," IROS, 2018. DOI: 10.1109/IROS.2018.8593739
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

# Reduced prediction state: [x, y, z, vx, vy, vz, phi, theta, psi]
_NX = 9


class NMPCTracker:
    """Nonlinear MPC using single-shooting with RK2 integration.

    The prediction model is

    .. code-block:: text

        ṗ = v
        v̇ = (T/m) R(η) e₃ - g e₃
        η̇ = W(η) ω

    with decision variables ``[T, ωx, ωy, ωz]`` per control block. The
    commanded body rates are closed by a proportional rate loop to produce
    the torques actually returned.

    Parameters
    ----------
    horizon : prediction steps.
    dt : prediction step [s] — how far ahead each predicted step reaches,
        not how often the controller runs.
    n_blocks : number of independent control moves over the horizon
        (move blocking). ``None`` gives one move per prediction step.
    mass, gravity : quadrotor parameters.
    inertia : 3x3 inertia tensor, used by the inner rate loop.
    Q : (9,) or (9,9) state cost over ``[p, v, η]`` (diagonal if 1-D).
    Qf : terminal state cost, defaults to ``5 Q``.
    R : (4,) or (4,4) cost on the deviation from hover input.
    max_thrust_ratio : maximum thrust as multiple of hover.
    max_rate : body-rate command clamp [rad/s].
    rate_gain : proportional gain of the inner body-rate loop [1/s].
    max_torque : torque clamp [Nm] on the returned wrench.
    max_iter : optimiser iteration cap per control step.
    """

    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.05,
        n_blocks: int | None = 4,
        mass: float = 1.5,
        gravity: float = 9.81,
        inertia: NDArray[np.floating] | None = None,
        Q: NDArray[np.floating] | None = None,
        Qf: NDArray[np.floating] | None = None,
        R: NDArray[np.floating] | None = None,
        max_thrust_ratio: float = 2.0,
        max_rate: float = 3.0,
        rate_gain: float = 25.0,
        max_torque: float = 1.0,
        max_iter: int = 20,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.n_blocks = horizon if n_blocks is None else min(n_blocks, horizon)
        self.mass = mass
        self.gravity = gravity
        self.inertia = inertia if inertia is not None else np.diag([0.0082, 0.0082, 0.0148])
        self.rate_gain = rate_gain
        self.max_iter = max_iter

        default_q = np.diag([16.0, 16.0, 24.0, 5.0, 5.0, 6.0, 0.4, 0.4, 0.4])
        self.Q = (
            np.diag(Q) if Q is not None and Q.ndim == 1 else (Q if Q is not None else default_q)
        )
        self.Qf = (
            np.diag(Qf)
            if Qf is not None and Qf.ndim == 1
            else (Qf if Qf is not None else 5.0 * self.Q)
        )
        self.R = (
            np.diag(R)
            if R is not None and R.ndim == 1
            else (R if R is not None else np.diag([0.002, 0.4, 0.4, 0.4]))
        )

        self.hover_input = np.array([mass * gravity, 0.0, 0.0, 0.0])
        self.max_thrust = mass * gravity * max_thrust_ratio
        self.max_rate = max_rate
        self.max_torque = max_torque
        # Which control block drives each prediction step.
        self._block_of_step = np.floor(np.arange(horizon) * self.n_blocks / horizon).astype(int)
        self._warm: NDArray[np.floating] | None = None
        self._last_plan: NDArray[np.floating] | None = None

    def reset(self) -> None:
        self._warm = None
        self._last_plan = None

    # ── reference handling ────────────────────────────────────────────────

    def _build_reference(
        self,
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """Expand the reference into one target per horizon step.

        A single point broadcast across the horizon tells the optimiser to
        arrive and stop where the reference is *now*, which on a moving
        trajectory costs roughly half a horizon of lag no matter how well
        the optimiser converges.
        """
        pos = np.atleast_2d(np.asarray(ref_pos, dtype=float))
        if len(pos) == 1:
            pos = np.repeat(pos, self.horizon, axis=0)
        vel = (
            np.zeros_like(pos)
            if ref_vel is None
            else np.atleast_2d(np.asarray(ref_vel, dtype=float))
        )
        if len(vel) == 1:
            vel = np.repeat(vel, self.horizon, axis=0)

        ref = np.zeros((self.horizon, _NX))
        ref[:, :3] = pos[: self.horizon]
        ref[:, 3:6] = vel[: self.horizon]
        return ref

    # ── main entry point ──────────────────────────────────────────────────

    def compute(
        self,
        state: NDArray[np.floating],
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Compute NMPC wrench.

        Parameters
        ----------
        state : 12-element state ``[x,y,z,φ,θ,ψ,vx,vy,vz,p,q,r]``.
        ref_pos : desired ``[x, y, z]``, or ``(horizon, 3)`` for a
            time-varying reference sampled at the prediction step.
        ref_vel : desired velocity, same shapes (default zeros).

        Returns
        -------
        ``[T, τx, τy, τz]`` wrench.
        """
        ref = self._build_reference(ref_pos, ref_vel)
        x0 = self._reduce(state)

        b = self.n_blocks
        if self._warm is not None and len(self._warm) == b * 4:
            u0 = self._warm.copy()
        else:
            u0 = np.tile(self.hover_input, b)

        bounds = []
        for _ in range(b):
            bounds.append((0.0, self.max_thrust))
            for _ in range(3):
                bounds.append((-self.max_rate, self.max_rate))

        result = minimize(
            self._cost,
            u0,
            args=(x0, ref),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-6},
        )
        blocks = result.x.reshape(b, 4)

        # Shift the block sequence by one for the next solve; repeat the
        # last block rather than resetting it to hover, which would throw
        # away most of the warm start on every step.
        self._warm = np.concatenate([result.x[4:], result.x[(b - 1) * 4 :]])
        self._last_plan = self._rollout(x0, blocks)

        return self._to_wrench(state, blocks[0])

    def predict(self) -> NDArray[np.floating] | None:
        """Predicted ``(horizon, 9)`` state trajectory from the last solve."""
        return None if self._last_plan is None else self._last_plan.copy()

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _reduce(state: NDArray[np.floating]) -> NDArray[np.floating]:
        """``[x,y,z,φ,θ,ψ,vx,vy,vz,...]`` → ``[p, v, η]``."""
        return np.concatenate([state[:3], state[6:9], state[3:6]])

    def _to_wrench(
        self,
        state: NDArray[np.floating],
        u: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Close the body-rate command with a proportional rate loop."""
        omega = np.asarray(state[9:12], dtype=float)
        omega_cmd = np.clip(u[1:], -self.max_rate, self.max_rate)
        inertia = np.asarray(self.inertia)
        tau = inertia @ (self.rate_gain * (omega_cmd - omega)) + np.cross(omega, inertia @ omega)
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        thrust = float(np.clip(u[0], 0.0, self.max_thrust))
        return np.array([thrust, tau[0], tau[1], tau[2]])

    def _cost(self, u_flat: NDArray, x0: NDArray, ref: NDArray) -> float:
        blocks = u_flat.reshape(self.n_blocks, 4)
        x = x0.copy()
        cost = 0.0
        for k in range(self.horizon):
            u = blocks[self._block_of_step[k]]
            x = self._step(x, u)
            e = x - ref[k]
            weight = self.Qf if k == self.horizon - 1 else self.Q
            du = u - self.hover_input
            cost += float(e @ weight @ e + du @ self.R @ du)
        return cost

    def _rollout(self, x0: NDArray, blocks: NDArray) -> NDArray[np.floating]:
        traj = np.zeros((self.horizon, _NX))
        x = x0.copy()
        for k in range(self.horizon):
            x = self._step(x, blocks[self._block_of_step[k]])
            traj[k] = x
        return traj

    def _step(self, state: NDArray, u: NDArray) -> NDArray:
        """RK2 integration of the reduced nonlinear model."""
        k1 = self._dynamics(state, u)
        k2 = self._dynamics(state + self.dt * k1, u)
        return state + 0.5 * self.dt * (k1 + k2)

    def _dynamics(self, state: NDArray, u: NDArray) -> NDArray:
        vx, vy, vz = state[3], state[4], state[5]
        phi, theta, psi = state[6], state[7], state[8]
        thrust = u[0]
        p, q, r = u[1], u[2], u[3]

        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)

        # Third column of the ZYX rotation matrix: body z-axis in world.
        Rz = np.array(
            [
                cy * st * cp + sy * sp,
                sy * st * cp - cy * sp,
                ct * cp,
            ]
        )
        acc = thrust / self.mass * Rz - np.array([0.0, 0.0, self.gravity])

        ct_safe = ct if abs(ct) > 1e-6 else np.sign(ct or 1.0) * 1e-6
        euler_dot = np.array(
            [
                p + (q * sp + r * cp) * (st / ct_safe),
                q * cp - r * sp,
                (q * sp + r * cp) / ct_safe,
            ]
        )

        ds = np.zeros(_NX)
        ds[:3] = [vx, vy, vz]
        ds[3:6] = acc
        ds[6:9] = euler_dot
        return ds
