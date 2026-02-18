# Erwin Lejeune - 2026-02-17
"""UKF localisation: multi-panel with position + velocity + innovation.

Shows step-by-step filter operation with true state, noisy measurements,
and UKF estimate converging, alongside innovation and covariance trace.

Reference: E. A. Wan, R. Van Der Merwe, "The Unscented Kalman Filter for
Nonlinear Estimation," AS-SPCC, 2000. DOI: 10.1109/ASSPCC.2000.882463
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.estimation.ukf import UnscentedKalmanFilter
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


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
    ukf.P = np.eye(2) * 0.5

    xt = np.array([0.0, 1.0])
    true_s = np.zeros((N, 2))
    est_s = np.zeros((N, 2))
    meas = np.zeros(N)
    innovation = np.zeros(N)
    cov_trace = np.zeros(N)

    for i in range(N):
        xt = _f(xt, None, dt) + rng.multivariate_normal([0, 0], Q * 0.1)
        z = _h(xt) + rng.multivariate_normal([0], R)
        ukf.predict(np.zeros(1), dt)
        inn = z[0] - _h(ukf.x)[0]
        ukf.update(z)
        true_s[i] = xt
        est_s[i] = ukf.x
        meas[i] = z[0]
        innovation[i] = inn
        cov_trace[i] = np.trace(ukf.P)

    t = np.arange(N) * dt
    skip = max(1, N // 200)
    idx = list(range(0, N, skip))
    n_frames = len(idx)

    anim = SimAnimator("ukf", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(12, 8))
    anim._fig = fig
    gs = fig.add_gridspec(3, 1, hspace=0.4)
    ax_pos = fig.add_subplot(gs[0])
    ax_inn = fig.add_subplot(gs[1])
    ax_cov = fig.add_subplot(gs[2])
    fig.suptitle("Unscented Kalman Filter — Step-by-step", fontsize=13)

    ax_pos.set_xlim(0, N * dt)
    ax_pos.set_ylim(true_s[:, 0].min() - 1, true_s[:, 0].max() + 1)
    ax_pos.set_ylabel("Position", fontsize=9)
    ax_pos.grid(True, alpha=0.3)
    (lt,) = ax_pos.plot([], [], "k-", lw=1.5, label="True")
    (le,) = ax_pos.plot([], [], "b-", lw=1.2, label="UKF")
    (sm,) = ax_pos.plot([], [], "r.", ms=3, alpha=0.4, label="Meas")
    ax_pos.legend(fontsize=7, ncol=3, loc="upper left")
    ax_pos.tick_params(labelsize=7)

    ax_inn.set_xlim(0, N * dt)
    inn_lim = max(1.0, np.abs(innovation).max() * 1.1)
    ax_inn.set_ylim(-inn_lim, inn_lim)
    ax_inn.set_ylabel("Innovation", fontsize=9)
    ax_inn.grid(True, alpha=0.3)
    ax_inn.axhline(0, color="k", ls=":", lw=0.5, alpha=0.4)
    (li,) = ax_inn.plot([], [], "m-", lw=1)
    ax_inn.tick_params(labelsize=7)

    ax_cov.set_xlim(0, N * dt)
    ax_cov.set_ylim(0, max(0.5, cov_trace.max() * 1.1))
    ax_cov.set_xlabel("Time [s]", fontsize=9)
    ax_cov.set_ylabel("tr(P)", fontsize=9)
    ax_cov.grid(True, alpha=0.3)
    (lc,) = ax_cov.plot([], [], "b-", lw=1)
    ax_cov.tick_params(labelsize=7)

    def update(f):
        k = idx[f]
        lt.set_data(t[:k], true_s[:k, 0])
        le.set_data(t[:k], est_s[:k, 0])
        sm.set_data(t[:k], meas[:k])
        li.set_data(t[:k], innovation[:k])
        lc.set_data(t[:k], cov_trace[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
