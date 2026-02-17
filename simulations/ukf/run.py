# Erwin Lejeune - 2026-02-15
"""UKF tracking a constant-velocity position with noisy measurements.

Reference: E. A. Wan, R. Van Der Merwe, "The Unscented Kalman Filter for
Nonlinear Estimation," AS-SPCC, 2000. DOI: 10.1109/ASSPCC.2000.882463
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.estimation.ukf import UnscentedKalmanFilter
from quadrotor_sim.visualization import SimAnimator


def _f(x, _u, dt):
    return np.array([x[0] + x[1] * dt, x[1]])


def _h(x):
    return x[:1]


def main() -> None:
    dt, N = 0.05, 200
    rng = np.random.default_rng(42)
    Q = np.diag([0.01, 0.1])
    R = np.array([[0.5]])
    ukf = UnscentedKalmanFilter(state_dim=2, meas_dim=1, f=_f, h=_h)
    ukf.Q = Q
    ukf.R = R
    ukf.x = np.array([0.0, 1.0])
    xt = np.array([0.0, 1.0])
    true_s = np.zeros((N, 2))
    est_s = np.zeros((N, 2))
    meas = np.zeros(N)
    for i in range(N):
        xt = _f(xt, None, dt) + rng.multivariate_normal([0, 0], Q * 0.1)
        z = _h(xt) + rng.multivariate_normal([0], R)
        ukf.predict(np.zeros(1), dt)
        ukf.update(z)
        true_s[i] = xt
        est_s[i] = ukf.x
        meas[i] = z[0]
    t = np.arange(N) * dt
    skip = max(1, N // 200)
    idx = list(range(0, N, skip))
    anim = SimAnimator("ukf", out_dir=Path(__file__).parent)
    _, axes = anim.figure_2d("Unscented Kalman Filter", nrows=1)
    ax = axes[0]
    ax.set_xlim(0, N * dt)
    ax.set_ylim(true_s[:, 0].min() - 1, true_s[:, 0].max() + 1)
    (lt,) = ax.plot([], [], "k-", lw=1.5, label="True")
    (le,) = ax.plot([], [], "b-", lw=1.2, label="UKF")
    (sm,) = ax.plot([], [], "r.", ms=3, alpha=0.4, label="Meas")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    def update(f):
        k = idx[f]
        lt.set_data(t[:k], true_s[:k, 0])
        le.set_data(t[:k], est_s[:k, 0])
        sm.set_data(t[:k], meas[:k])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
