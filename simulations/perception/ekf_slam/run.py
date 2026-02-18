# Erwin Lejeune - 2026-02-17
"""EKF-SLAM: quadrotor maps landmarks while localising itself.

The drone flies a circular pattern via pure pursuit, observing point-landmarks
with a range-bearing sensor.  The EKF jointly estimates the drone's 2D pose
and all landmark positions, showing covariance ellipses for both.

Reference: S. Thrun et al., "Probabilistic Robotics," MIT Press, 2005, Ch. 10.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
)

matplotlib.use("Agg")

# ── EKF-SLAM core ─────────────────────────────────────────────────────────
_N_LM = 8
_STATE_DIM = 3  # robot: [x, y, yaw]
_LM_DIM = 2  # each landmark: [lx, ly]

RANGE_STD = 0.3
BEARING_STD = 0.1
ODOM_V_STD = 0.2
ODOM_W_STD = 0.05


def _motion_model(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Simple constant-velocity unicycle model for the robot state."""
    v, w = u
    yaw = x[2]
    return x + np.array([v * np.cos(yaw) * dt, v * np.sin(yaw) * dt, w * dt])


def _range_bearing(rx: float, ry: float, ryaw: float, lx: float, ly: float):
    dx = lx - rx
    dy = ly - ry
    r = np.sqrt(dx**2 + dy**2)
    b = np.arctan2(dy, dx) - ryaw
    b = (b + np.pi) % (2 * np.pi) - np.pi
    return r, b


def main() -> None:
    rng = np.random.default_rng(42)

    # True landmark positions (scaled to match smaller flight radius)
    landmarks = np.array(
        [
            [2, 2],
            [-2, 2],
            [-2, -2],
            [2, -2],
            [3, 0],
            [0, 3],
            [-3, 0],
            [0, -3],
        ],
        dtype=float,
    )[:_N_LM]

    # ── Generate flight path (arc 0→270° at altitude 1, open-ended) ─────
    radius = 2.0
    cruise_alt = 1.0
    n_wp = 40
    angles = np.linspace(0, 1.5 * np.pi, n_wp)
    path_3d = np.column_stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(n_wp, cruise_alt),
        ]
    )

    quad = Quadrotor()
    quad.reset(position=np.array([radius, 0.0, cruise_alt]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=0.5, waypoint_threshold=0.2, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=60.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))

    # ── Run EKF-SLAM over the recorded trajectory ─────────────────────────
    n_steps = len(flight_states)
    dt = 0.005
    full_dim = _STATE_DIM + _N_LM * _LM_DIM
    mu = np.zeros(full_dim)
    mu[:2] = flight_states[0, :2]
    mu[2] = flight_states[0, 5]
    sigma = np.eye(full_dim) * 100.0
    sigma[:3, :3] = np.diag([0.01, 0.01, 0.01])
    seen = np.zeros(_N_LM, dtype=bool)

    robot_est_hist = np.zeros((n_steps, 3))
    robot_true_hist = np.zeros((n_steps, 2))
    lm_est_hist = np.zeros((n_steps, _N_LM, 2))
    lm_cov_hist = np.zeros((n_steps, _N_LM, 2, 2))

    for step in range(n_steps):
        s = flight_states[step]
        true_xy = s[:2]
        true_yaw = s[5]
        robot_true_hist[step] = true_xy

        # Predict (use velocity from state)
        if step > 0:
            v = float(np.linalg.norm(s[6:8]))
            w = float(s[11]) if len(s) > 11 else 0.0
            u = np.array([v + rng.normal(0, ODOM_V_STD), w + rng.normal(0, ODOM_W_STD)])
            mu[:3] = _motion_model(mu[:3], u, dt)
            F = np.eye(full_dim)
            yaw = mu[2]
            F[0, 2] = -u[0] * np.sin(yaw) * dt
            F[1, 2] = u[0] * np.cos(yaw) * dt
            Q = np.zeros((full_dim, full_dim))
            Q[:3, :3] = np.diag([ODOM_V_STD**2 * dt, ODOM_V_STD**2 * dt, ODOM_W_STD**2 * dt])
            sigma = F @ sigma @ F.T + Q

        # Update with landmark observations
        for j in range(_N_LM):
            r_true, b_true = _range_bearing(
                true_xy[0], true_xy[1], true_yaw, landmarks[j, 0], landmarks[j, 1]
            )
            if r_true > 5.0:
                continue
            r_meas = r_true + rng.normal(0, RANGE_STD)
            b_meas = b_true + rng.normal(0, BEARING_STD)

            idx = _STATE_DIM + j * _LM_DIM
            if not seen[j]:
                mu[idx] = mu[0] + r_meas * np.cos(mu[2] + b_meas)
                mu[idx + 1] = mu[1] + r_meas * np.sin(mu[2] + b_meas)
                seen[j] = True

            dx = mu[idx] - mu[0]
            dy = mu[idx + 1] - mu[1]
            q = dx**2 + dy**2
            sq = max(np.sqrt(q), 1e-6)
            r_pred = sq
            b_pred = np.arctan2(dy, dx) - mu[2]
            b_pred = (b_pred + np.pi) % (2 * np.pi) - np.pi

            H = np.zeros((2, full_dim))
            H[0, 0] = -dx / sq
            H[0, 1] = -dy / sq
            H[1, 0] = dy / q
            H[1, 1] = -dx / q
            H[1, 2] = -1.0
            H[0, idx] = dx / sq
            H[0, idx + 1] = dy / sq
            H[1, idx] = -dy / q
            H[1, idx + 1] = dx / q

            R = np.diag([RANGE_STD**2, BEARING_STD**2])
            S = H @ sigma @ H.T + R
            K = sigma @ H.T @ np.linalg.inv(S)
            innov = np.array([r_meas - r_pred, b_meas - b_pred])
            innov[1] = (innov[1] + np.pi) % (2 * np.pi) - np.pi
            mu = mu + K @ innov
            sigma = (np.eye(full_dim) - K @ H) @ sigma

        robot_est_hist[step] = mu[:3]
        for j in range(_N_LM):
            idx = _STATE_DIM + j * _LM_DIM
            lm_est_hist[step, j] = mu[idx : idx + 2]
            lm_cov_hist[step, j] = sigma[idx : idx + 2, idx : idx + 2]

    # ── Animation ─────────────────────────────────────────────────────────
    skip = max(1, n_steps // 200)
    frames = list(range(0, n_steps, skip))
    n_frames = len(frames)

    anim = SimAnimator("ekf_slam", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(10, 8))
    anim._fig = fig
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.2)
    fig.suptitle("EKF-SLAM — Landmark Mapping", fontsize=13)

    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        c="red",
        s=100,
        marker="*",
        zorder=10,
        label="True Landmarks",
    )
    (true_trail,) = ax.plot([], [], "k-", lw=1.0, alpha=0.5, label="True path")
    (est_trail,) = ax.plot([], [], "b-", lw=1.0, alpha=0.8, label="EKF estimate")

    lm_scats = ax.scatter([], [], c="blue", s=60, marker="D", zorder=8, label="Est. LMs")
    ellipses: list[Ellipse] = []
    for _ in range(_N_LM):
        e = Ellipse((0, 0), 0, 0, fill=False, color="blue", lw=0.8, ls="--", alpha=0.5)
        ax.add_patch(e)
        ellipses.append(e)

    ax.legend(fontsize=7, loc="upper left")
    vehicle_arts: list = []

    def _update_ellipse(patch, mean, cov, n_std=2.0):
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-8)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        patch.set_center(mean)
        patch.width = 2 * n_std * np.sqrt(vals[0])
        patch.height = 2 * n_std * np.sqrt(vals[1])
        patch.angle = angle

    def update(f):
        k = frames[f]
        true_trail.set_data(robot_true_hist[:k, 0], robot_true_hist[:k, 1])
        est_trail.set_data(robot_est_hist[:k, 0], robot_est_hist[:k, 1])

        lm_pts = lm_est_hist[k]
        vis = seen.copy()
        lm_scats.set_offsets(lm_pts[vis])
        for j in range(_N_LM):
            if vis[j]:
                _update_ellipse(ellipses[j], lm_pts[j], lm_cov_hist[k, j])
                ellipses[j].set_visible(True)
            else:
                ellipses[j].set_visible(False)

        clear_vehicle_artists(vehicle_arts)
        vehicle_arts.extend(
            draw_quadrotor_2d(ax, robot_true_hist[k], flight_states[k, 5], size=0.4),
        )

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
