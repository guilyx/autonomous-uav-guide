"""Shared visual-servoing runner for gimbal and fixed camera variants."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.environment import default_world
from uav_sim.logging import SimLogger
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.perception.bbox_tracker import (
    SimulatedDetector,
    VisualServoConfig,
    VisualServoController,
)
from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import PointTracker
from uav_sim.simulations.standards import SimulationStandard
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")

WORLD_SIZE = 30.0
H_FOV = 0.8
V_FOV = 0.6


def _arch_target(t: float, duration: float) -> np.ndarray:
    progress = np.clip(t / max(duration, 1e-6), 0.0, 1.0)
    x = 4.0 + 22.0 * progress
    y = 15.0 + 7.5 * np.sin(np.pi * progress)
    z = 0.6 + 0.4 * np.sin(2.0 * np.pi * progress)
    return np.array([x, y, z])


def run_visual_servoing(
    *,
    sim_name: str,
    out_dir: Path,
    gimbal_tracking: bool,
) -> None:
    _, buildings = default_world()
    standard = replace(SimulationStandard.trajectory_tracking(), dt=0.02, duration=60.0)
    n_steps = int(standard.duration / standard.dt)

    gimbal = Gimbal(max_rate=2.5)
    init_target = _arch_target(0.0, standard.duration)
    init_pos = np.array([6.0, 11.0, 8.0])
    init_pan, init_tilt = gimbal.look_at(init_pos, init_target, 0.0)
    if gimbal_tracking:
        gimbal.reset(pan=init_pan, tilt=init_tilt)
    else:
        # Fixed camera mode: lock pan to body-forward, keep a realistic down tilt.
        gimbal.reset(pan=0.0, tilt=float(init_tilt))
    tracker = PointTracker(gimbal) if gimbal_tracking else None

    detector = SimulatedDetector(target_radius=0.55, ndc_noise_std=0.01, seed=42)
    servo_cfg = (
        VisualServoConfig(
            kp_lateral=2.2,
            kp_forward=3.2,
            desired_size_ratio=0.10,
            max_velocity=3.0,
        )
        if gimbal_tracking
        else VisualServoConfig(
            kp_lateral=1.7,
            kp_forward=2.1,
            desired_size_ratio=0.09,
            max_velocity=2.3,
        )
    )
    servo = VisualServoController(servo_cfg)

    quad = Quadrotor()
    quad.reset(position=init_pos.copy())
    ctrl = CascadedPIDController()
    follow_setpoint = init_pos.copy()

    drone_pos = np.zeros((n_steps, 3))
    drone_att = np.zeros((n_steps, 3))
    target_pos = np.zeros((n_steps, 3))
    bbox_cx = np.zeros(n_steps)
    bbox_cy = np.zeros(n_steps)
    bbox_size = np.zeros(n_steps)
    visible_hist = np.zeros(n_steps, dtype=bool)
    dist_hist = np.zeros(n_steps)
    center_err = np.zeros(n_steps)
    size_err = np.zeros(n_steps)

    for i in range(n_steps):
        t = i * standard.dt
        target = _arch_target(t, standard.duration)
        target_pos[i] = target

        state = quad.state.copy()
        pos = state[:3]
        yaw = state[5]
        desired_yaw = 0.0
        if tracker is not None:
            tracker.step(pos, target, yaw, standard.dt)
        else:
            to_target = target - pos
            desired_yaw = float(np.arctan2(to_target[1], to_target[0]))

        det = detector.detect(target, pos, gimbal, H_FOV, V_FOV, yaw)
        vel_cmd = servo.compute(det, yaw)
        if not gimbal_tracking and not det.visible:
            # If target exits frame, rotate to reacquire and move gently toward target.
            vel_cmd = 0.35 * np.array([np.cos(desired_yaw), np.sin(desired_yaw), -0.10])
        follow_setpoint = follow_setpoint + vel_cmd * standard.dt
        follow_setpoint[0] = np.clip(follow_setpoint[0], 1.0, WORLD_SIZE - 1.0)
        follow_setpoint[1] = np.clip(follow_setpoint[1], 1.0, WORLD_SIZE - 1.0)
        follow_setpoint[2] = np.clip(follow_setpoint[2], 3.0, 18.0)

        wrench = ctrl.compute(quad.state, follow_setpoint, target_yaw=desired_yaw, dt=standard.dt)
        quad.step(wrench, standard.dt)

        new_state = quad.state.copy()
        drone_pos[i] = new_state[:3]
        drone_att[i] = new_state[3:6]
        visible_hist[i] = det.visible
        bbox_cx[i] = det.center_ndc[0] if det.visible else 0.0
        bbox_cy[i] = det.center_ndc[1] if det.visible else 0.0
        bbox_size[i] = det.size_ratio if det.visible else 0.0
        dist_hist[i] = float(np.linalg.norm(new_state[:3] - target))
        center_err[i] = float(np.linalg.norm(det.center_ndc)) if det.visible else 1.5
        size_err[i] = abs(servo.cfg.desired_size_ratio - det.size_ratio) if det.visible else 1.0

    logger = SimLogger(sim_name, out_dir=out_dir)
    logger.log_metadata("algorithm", "Visual Servoing")
    logger.log_metadata("target_motion", "arch")
    logger.log_metadata("camera_mode", "gimbal" if gimbal_tracking else "fixed")
    logger.log_metadata("flight_coupled", False)
    logger.log_metadata("dt", standard.dt)
    logger.log_metadata("n_steps", n_steps)
    for i in range(n_steps):
        logger.log_step(
            t=i * standard.dt,
            drone_position=drone_pos[i].tolist(),
            drone_attitude=drone_att[i].tolist(),
            target_position=target_pos[i].tolist(),
            distance=float(dist_hist[i]),
            bbox_center_error=float(center_err[i]),
            bbox_size_error=float(size_err[i]),
            visible=bool(visible_hist[i]),
        )
    logger.log_summary("visibility_pct", float(100.0 * visible_hist.mean()))
    logger.log_summary("mean_distance_m", float(dist_hist.mean()))
    logger.log_summary("mean_bbox_center_error", float(center_err.mean()))
    logger.log_summary("mean_bbox_size_error", float(size_err.mean()))
    logger.save()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.30)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_cam = fig.add_subplot(gs[1, 0])
    ax_data = fig.add_subplot(gs[1, 1])
    mode = "Gimbal Camera" if gimbal_tracking else "Fixed Camera"
    fig.suptitle(f"Visual Servoing - {mode} (Arch Target)", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_top.set_xlim(0, WORLD_SIZE)
    ax_top.set_ylim(0, WORLD_SIZE)
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.set_title("Top Down", fontsize=9)
    ax_top.set_aspect("equal")
    for b in buildings:
        sz = b.max_corner - b.min_corner
        ax_top.add_patch(
            matplotlib.patches.Rectangle(b.min_corner[:2], sz[0], sz[1], fc="gray", alpha=0.3)
        )

    ax_top.plot(target_pos[:, 0], target_pos[:, 1], "r--", lw=0.7, alpha=0.4)
    (tgt_top,) = ax_top.plot([], [], "r*", ms=10, zorder=10)
    (tgt_3d,) = ax3d.plot([], [], [], "r*", ms=10, zorder=10, label="Target")
    (drone_trail_top,) = ax_top.plot([], [], "dodgerblue", lw=0.9, alpha=0.6)
    ax3d.legend(fontsize=7, loc="upper left")

    ax_cam.set_xlim(-1.2, 1.2)
    ax_cam.set_ylim(-1.0, 1.0)
    ax_cam.set_facecolor("black")
    ax_cam.set_title("Camera View", fontsize=9, color="white")
    ax_cam.tick_params(labelsize=5, colors="white")
    ax_cam.axhline(0, color="gray", lw=0.3, alpha=0.5)
    ax_cam.axvline(0, color="gray", lw=0.3, alpha=0.5)
    (bbox_rect_art,) = ax_cam.plot([], [], "lime", lw=2)
    ax_cam.plot([0], [0], "w+", ms=12, mew=1)
    (target_dot,) = ax_cam.plot([], [], "r.", ms=10)

    times = np.arange(n_steps) * standard.dt
    ax_data.set_xlim(0, standard.duration)
    ax_data.set_ylim(0, max(1.0, max(center_err.max(), size_err.max()) * 1.2))
    ax_data.set_xlabel("Time [s]", fontsize=8)
    ax_data.set_ylabel("Objective Error", fontsize=8)
    ax_data.set_title("BB Center/Size Tracking Error", fontsize=9)
    ax_data.grid(True, alpha=0.3)
    (center_line,) = ax_data.plot([], [], "c-", lw=0.8, label="Center error")
    (size_line,) = ax_data.plot([], [], "m--", lw=0.8, label="Size error")
    ax_data.legend(fontsize=7)

    skip = max(1, n_steps // 220)
    frames = list(range(0, n_steps, skip))
    veh_arts: list = []

    anim = SimAnimator(sim_name, out_dir=out_dir, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        k = frames[f]
        dp = drone_pos[k]
        tp = target_pos[k]

        drone_trail_top.set_data(drone_pos[: k + 1, 0], drone_pos[: k + 1, 1])
        clear_vehicle_artists(veh_arts)
        R = Quadrotor.rotation_matrix(*drone_att[k])
        veh_arts.extend(draw_quadrotor_3d(ax3d, dp, R, size=1.5))
        (dtop,) = ax_top.plot(dp[0], dp[1], "ko", ms=4, zorder=10)
        veh_arts.append(dtop)

        tgt_3d.set_data([tp[0]], [tp[1]])
        tgt_3d.set_3d_properties([tp[2]])
        tgt_top.set_data([tp[0]], [tp[1]])

        if visible_hist[k]:
            cx = bbox_cx[k]
            cy = bbox_cy[k]
            half = max(0.05, bbox_size[k] * 2.0)
            xs = [cx - half, cx + half, cx + half, cx - half, cx - half]
            ys = [cy - half, cy - half, cy + half, cy + half, cy - half]
            bbox_rect_art.set_data(xs, ys)
            target_dot.set_data([cx], [cy])
        else:
            bbox_rect_art.set_data([], [])
            target_dot.set_data([], [])

        center_line.set_data(times[: k + 1], center_err[: k + 1])
        size_line.set_data(times[: k + 1], size_err[: k + 1])

    anim.animate(update, len(frames))
    anim.save()
