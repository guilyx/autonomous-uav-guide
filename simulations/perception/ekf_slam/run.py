# Erwin Lejeune - 2026-02-18
"""EKF-SLAM: quadrotor maps landmarks while localising itself.

Three panels (3D, top-down, side) + insets:
  - 3D scene with quadrotor, true landmarks, estimated landmarks.
  - Top-down with covariance ellipses, FOV wedge, range-bearing lines.
  - Insets: position error and landmark discovery count.

The landmarks are progressively discovered as the drone orbits the world.

Reference: S. Thrun et al., "Probabilistic Robotics," MIT Press, 2005, Ch. 10.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Ellipse, Wedge

from uav_sim.path_tracking.flight_ops import fly_path, init_hover
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists

matplotlib.use("Agg")

_N_LM = 8
_STATE_DIM = 3
_LM_DIM = 2

RANGE_STD = 0.3
BEARING_STD = 0.1
ODOM_V_STD = 0.15
ODOM_W_STD = 0.04
SENSOR_MAX_RANGE = 18.0
SENSOR_FOV = np.radians(270.0)

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def _motion_model(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
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

    cx, cy = 15.0, 15.0
    landmarks = np.array(
        [
            [cx + 8, cy + 8],
            [cx - 8, cy + 8],
            [cx - 8, cy - 8],
            [cx + 8, cy - 8],
            [cx + 12, cy],
            [cx, cy + 12],
            [cx - 12, cy],
            [cx, cy - 12],
        ],
        dtype=float,
    )[:_N_LM]
    lm_z = np.full(_N_LM, 5.0)

    # Full circle + extra overlap for maximum landmark coverage
    radius = 10.0
    n_wp = 80
    angles = np.linspace(0, 2.2 * np.pi, n_wp)
    path_3d = np.column_stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles), np.full(n_wp, CRUISE_ALT)]
    )

    quad = Quadrotor()
    quad.reset(position=np.array([cx + radius, cy, CRUISE_ALT]))
    init_hover(quad)
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=80.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))

    # ── EKF-SLAM at reduced rate for efficiency ────────────────────────
    n_raw = len(flight_states)
    slam_dt = 0.02
    slam_skip = max(1, int(slam_dt / 0.005))
    slam_states = flight_states[::slam_skip]
    n_steps = len(slam_states)

    full_dim = _STATE_DIM + _N_LM * _LM_DIM
    mu = np.zeros(full_dim)
    mu[:2] = slam_states[0, :2]
    mu[2] = slam_states[0, 5]
    sigma = np.eye(full_dim) * 100.0
    sigma[:3, :3] = np.diag([0.01, 0.01, 0.01])
    seen = np.zeros(_N_LM, dtype=bool)

    robot_est_hist = np.zeros((n_steps, 3))
    robot_true_hist = np.zeros((n_steps, 2))
    lm_est_hist = np.zeros((n_steps, _N_LM, 2))
    lm_cov_hist = np.zeros((n_steps, _N_LM, 2, 2))
    seen_hist = np.zeros((n_steps, _N_LM), dtype=bool)
    n_seen_hist = np.zeros(n_steps, dtype=int)
    pos_err_hist = np.zeros(n_steps)
    obs_record: list[list[int]] = []

    for step in range(n_steps):
        s = slam_states[step]
        true_xy = s[:2]
        true_yaw = s[5]
        robot_true_hist[step] = true_xy

        if step > 0:
            v = float(np.linalg.norm(s[6:8]))
            w = float(s[11]) if len(s) > 11 else 0.0
            u = np.array([v + rng.normal(0, ODOM_V_STD), w + rng.normal(0, ODOM_W_STD)])
            mu[:3] = _motion_model(mu[:3], u, slam_dt)
            F = np.eye(full_dim)
            yaw = mu[2]
            F[0, 2] = -u[0] * np.sin(yaw) * slam_dt
            F[1, 2] = u[0] * np.cos(yaw) * slam_dt
            Q = np.zeros((full_dim, full_dim))
            Q[:3, :3] = np.diag(
                [ODOM_V_STD**2 * slam_dt, ODOM_V_STD**2 * slam_dt, ODOM_W_STD**2 * slam_dt]
            )
            sigma = F @ sigma @ F.T + Q

        observed: list[int] = []
        for j in range(_N_LM):
            r_true, b_true = _range_bearing(
                true_xy[0], true_xy[1], true_yaw, landmarks[j, 0], landmarks[j, 1]
            )
            if r_true > SENSOR_MAX_RANGE or abs(b_true) > SENSOR_FOV / 2:
                continue
            observed.append(j)
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
            r_pred = sq
            b_pred = np.arctan2(dy, dx) - mu[2]
            b_pred = (b_pred + np.pi) % (2 * np.pi) - np.pi
            innov = np.array([r_meas - r_pred, b_meas - b_pred])
            innov[1] = (innov[1] + np.pi) % (2 * np.pi) - np.pi
            mu = mu + K @ innov
            sigma = (np.eye(full_dim) - K @ H) @ sigma

        obs_record.append(observed)
        robot_est_hist[step] = mu[:3]
        for j in range(_N_LM):
            idx = _STATE_DIM + j * _LM_DIM
            lm_est_hist[step, j] = mu[idx : idx + 2]
            lm_cov_hist[step, j] = sigma[idx : idx + 2, idx : idx + 2]
        seen_hist[step] = seen.copy()
        n_seen_hist[step] = int(np.sum(seen))
        pos_err_hist[step] = np.linalg.norm(true_xy - mu[:2])

    # ── 3-Panel viz + insets ──────────────────────────────────────────
    viz = ThreePanelViz(
        title="EKF-SLAM — Landmark Mapping", world_size=WORLD_SIZE, figsize=(18, 9)
    )

    viz.ax3d.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        lm_z,
        c="red",
        s=80,
        marker="*",
        zorder=10,
        label="True LMs",
    )
    viz.ax3d.legend(fontsize=6, loc="upper left")
    viz.ax_top.scatter(
        landmarks[:, 0], landmarks[:, 1], c="red", s=80, marker="*", zorder=10, label="True LMs"
    )
    lm_scats = viz.ax_top.scatter([], [], c="blue", s=40, marker="D", zorder=8, label="Est. LMs")

    ellipses: list[Ellipse] = []
    for _ in range(_N_LM):
        e = Ellipse((0, 0), 0, 0, fill=False, color="blue", lw=0.8, ls="--", alpha=0.5)
        viz.ax_top.add_patch(e)
        e.set_visible(False)
        ellipses.append(e)
    viz.ax_top.legend(fontsize=6, loc="upper left")

    anim = SimAnimator("ekf_slam", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_true = viz.create_trail_artists(color="black")
    trail_est = viz.create_trail_artists(color="dodgerblue")

    times = np.arange(n_steps) * slam_dt
    ax_err = viz.fig.add_axes([0.60, 0.03, 0.36, 0.12])
    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(0, max(2.0, pos_err_hist.max() * 1.2))
    ax_err.set_xlabel("Time [s]", fontsize=6)
    ax_err.set_ylabel("Pos err [m]", fontsize=6)
    ax_err.tick_params(labelsize=5)
    ax_err.grid(True, alpha=0.2)
    (err_line,) = ax_err.plot([], [], "r-", lw=0.7)
    ax_lm = ax_err.twinx()
    ax_lm.set_ylim(0, _N_LM + 1)
    ax_lm.set_ylabel("# LMs", fontsize=6, color="seagreen")
    ax_lm.tick_params(labelsize=5, colors="seagreen")
    (lm_line,) = ax_lm.plot([], [], "seagreen", lw=0.7)

    skip = max(1, n_steps // 200)
    frames = list(range(0, n_steps, skip))
    n_frames_anim = len(frames)

    fov_arts: list = []
    obs_arts: list = []
    lm_3d_arts: list = []

    true_3d = np.column_stack([robot_true_hist, np.full(n_steps, CRUISE_ALT)])
    est_3d = np.column_stack([robot_est_hist[:, :2], np.full(n_steps, CRUISE_ALT)])

    # Map from slam step to flight_states index for euler angles
    slam_to_flight = [min(i * slam_skip, n_raw - 1) for i in range(n_steps)]

    def _update_ellipse(patch, mean, cov, n_std=2.0):
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-8)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        patch.set_center(mean)
        patch.width = 2 * n_std * np.sqrt(vals[0])
        patch.height = 2 * n_std * np.sqrt(vals[1])
        patch.angle = angle

    def update(f: int) -> None:
        k = frames[f]
        fi = slam_to_flight[k]

        viz.update_trail(trail_true, true_3d, k)
        viz.update_trail(trail_est, est_3d, k)
        viz.update_vehicle(true_3d[k], flight_states[fi, 3:6], size=1.5)

        vis_k = seen_hist[k]
        lm_pts = lm_est_hist[k]
        vis_pts = lm_pts[vis_k]
        lm_scats.set_offsets(vis_pts if len(vis_pts) > 0 else np.empty((0, 2)))

        for j in range(_N_LM):
            if vis_k[j]:
                _update_ellipse(ellipses[j], lm_pts[j], lm_cov_hist[k, j])
                ellipses[j].set_visible(True)
            else:
                ellipses[j].set_visible(False)

        clear_vehicle_artists(lm_3d_arts)
        if len(vis_pts) > 0:
            sc = viz.ax3d.scatter(
                vis_pts[:, 0],
                vis_pts[:, 1],
                np.full(len(vis_pts), 5.0),
                c="blue",
                s=30,
                marker="D",
                zorder=8,
            )
            lm_3d_arts.append(sc)

        clear_vehicle_artists(fov_arts)
        clear_vehicle_artists(obs_arts)
        yaw_k = flight_states[fi, 5]
        true_pos = robot_true_hist[k]

        fov_start = np.degrees(yaw_k - SENSOR_FOV / 2)
        wedge = Wedge(
            true_pos,
            SENSOR_MAX_RANGE,
            fov_start,
            fov_start + np.degrees(SENSOR_FOV),
            color="gold",
            alpha=0.06,
            lw=0,
        )
        viz.ax_top.add_patch(wedge)
        fov_arts.append(wedge)

        for j in obs_record[min(k, len(obs_record) - 1)]:
            (ln,) = viz.ax_top.plot(
                [true_pos[0], landmarks[j, 0]],
                [true_pos[1], landmarks[j, 1]],
                "gold",
                lw=0.7,
                alpha=0.4,
            )
            obs_arts.append(ln)

        err_line.set_data(times[:k], pos_err_hist[:k])
        lm_line.set_data(times[:k], n_seen_hist[:k])

    anim.animate(update, n_frames_anim)
    anim.save()


if __name__ == "__main__":
    main()
