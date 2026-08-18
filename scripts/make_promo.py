# Erwin Lejeune - 2026-02-19
"""Render the promotional video.

Everything on screen is simulated live by this script — there are no
pre-rendered clips and no stock footage. The trajectories you see are the
same models the library ships, integrated at render time, which means the
video cannot drift out of sync with the code.

    uv run python scripts/make_promo.py --output media/promo.mp4

Requires the ``video`` extra for the ffmpeg binary::

    uv pip install imageio-ffmpeg
"""

from __future__ import annotations

import argparse

# Only ever used to pipe raw frames into the ffmpeg that imageio-ffmpeg bundles.
import subprocess  # nosec B404
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# ── palette ───────────────────────────────────────────────────────────────
# Mirrors docs/.vitepress/theme/style.css so the video, the site and the
# CLI read as one product.

INK = "#070b14"
PANEL = "#0f172a"
GRID = "#1e293b"
TEXT = "#e6edf7"
MUTED = "#94a3b8"
SKY = "#38bdf8"
TEAL = "#2dd4bf"
AMBER = "#fbbf24"
VIOLET = "#a78bfa"
ROSE = "#fb7185"

TRACE_COLOURS = [SKY, TEAL, VIOLET, AMBER, ROSE]

FPS = 30
# x264 quality. The video is committed to docs/public/ rather than LFS, so
# it has to stay under the repository's large-file guard: the atlas scene's
# forty-two plots are fine detail on a flat background, and cost several
# times the bitrate of the vector-art scenes around them.
CRF = 23
# Index into build_scenes() output: the multirotor scene.
POSTER_SCENE = 1
WIDTH, HEIGHT = 1920, 1080
DPI = 120

# Framing shared by the scenes that are one 3D plot and nothing else. The
# zoom is as large as it goes before the projected cube's tick labels reach
# the caption line.
FULL_3D_RECT = [0.08, 0.19, 0.84, 0.64]
FULL_3D_ZOOM = 1.3


@dataclass
class Scene:
    """One segment of the video."""

    title: str
    subtitle: str
    seconds: float
    draw: object
    """``draw(figure, progress)`` — progress runs 0 → 1 across the scene."""
    caption: str = ""
    stats: list[tuple[str, str]] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════
# Shared drawing helpers
# ══════════════════════════════════════════════════════════════════════════


def _style_3d(axes, limits, *, elev=22, azim=-60, zoom=1.0):
    """Style a 3D axes and fit the projected cube to its rectangle.

    ``zoom`` scales the cube inside the axes. A 3D projection is drawn
    into a square region regardless of how wide its rectangle is, so a
    full-width scene otherwise leaves a third of the frame empty on each
    side. Panelled scenes keep the default.
    """
    axes.set_facecolor(INK)
    axes.set_xlim(limits[0])
    axes.set_ylim(limits[1])
    axes.set_zlim(limits[2])
    axes.view_init(elev=elev, azim=azim)
    if zoom != 1.0:
        axes.set_box_aspect(axes.get_box_aspect(), zoom=zoom)
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color((0.03, 0.05, 0.09, 1.0))
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.5
    axes.tick_params(colors=MUTED, labelsize=7)
    axes.set_xlabel("x [m]", color=MUTED, fontsize=8, labelpad=-4)
    axes.set_ylabel("y [m]", color=MUTED, fontsize=8, labelpad=-4)
    axes.set_zlabel("z [m]", color=MUTED, fontsize=8, labelpad=-4)


def _draw_chrome(figure, scene: Scene, progress: float):
    """Title block, caption, stat strip and progress rule."""
    figure.text(
        0.055,
        0.915,
        scene.title,
        color=TEXT,
        fontsize=30,
        fontweight="bold",
        va="top",
        family="DejaVu Sans",
    )
    figure.text(
        0.055,
        0.868,
        scene.subtitle,
        color=SKY,
        fontsize=14,
        va="top",
        family="DejaVu Sans Mono",
    )
    if scene.caption:
        figure.text(
            0.055,
            0.075,
            scene.caption,
            color=MUTED,
            fontsize=13,
            va="bottom",
            family="DejaVu Sans",
        )

    # Stat strip, bottom right.
    for index, (value, label) in enumerate(scene.stats):
        x = 0.955 - (len(scene.stats) - 1 - index) * 0.135
        figure.text(
            x,
            0.115,
            value,
            color=SKY,
            fontsize=22,
            fontweight="bold",
            ha="right",
            family="DejaVu Sans Mono",
        )
        figure.text(
            x,
            0.075,
            label,
            color=MUTED,
            fontsize=9,
            ha="right",
            family="DejaVu Sans Mono",
        )

    # Progress rule along the bottom edge.
    figure.add_artist(matplotlib.lines.Line2D([0.0, 1.0], [0.012, 0.012], color=GRID, lw=2.5))
    figure.add_artist(
        matplotlib.lines.Line2D(
            [0.0, float(np.clip(progress, 0.0, 1.0))], [0.012, 0.012], color=SKY, lw=2.5
        )
    )


def _fade(figure, alpha: float):
    """Overlay a full-frame scrim, for cross-fades between scenes."""
    if alpha <= 0.001:
        return
    figure.patches.append(
        matplotlib.patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=figure.transFigure,
            facecolor=INK,
            alpha=float(np.clip(alpha, 0, 1)),
            zorder=1000,
        )
    )


# ══════════════════════════════════════════════════════════════════════════
# Simulations — run once up front, then replayed frame by frame
# ══════════════════════════════════════════════════════════════════════════


def simulate_quadrotor():
    """Figure-eight tracking under the geometric SO(3) controller.

    Uses the geometric controller rather than the cascaded PID because it
    accepts velocity and acceleration feed-forward. A pure feedback
    cascade lags a moving reference in proportion to its speed — perfectly
    correct behaviour, and not what a trajectory-tracking demo should be
    showing.
    """
    from uav_sim.path_tracking.geometric_controller import GeometricController
    from uav_sim.vehicles.multirotor import Quadrotor

    quad = Quadrotor()
    controller = GeometricController()

    dt = 0.005
    steps = 8000
    states = np.zeros((steps, 12))
    reference = np.zeros((steps, 3))

    amplitude = np.array([5.0, 4.0, 1.0])
    omega = np.array([0.32, 0.64, 0.24])

    # Start on the trajectory, moving at its velocity. Starting from rest
    # spends the first second of the clip on a transient that says nothing
    # about the controller.
    quad.reset(
        position=np.array([0.0, 0.0, 3.0]),
        velocity=amplitude * omega,
    )

    for index in range(steps):
        t = index * dt
        phase = omega * t
        target = np.array([0.0, 0.0, 3.0]) + amplitude * np.sin(phase)
        velocity = amplitude * omega * np.cos(phase)
        acceleration = -amplitude * omega**2 * np.sin(phase)

        reference[index] = target
        states[index] = quad.state
        # dt lets the controller differentiate R_d for the desired
        # angular velocity; without it that feed-forward is assumed zero.
        quad.step(controller.compute(quad.state, target, velocity, acceleration, dt=dt), dt)

    return states, reference


def simulate_fixed_wing():
    """Autopilot climbing and turning onto a new course."""
    from uav_sim.control.fixed_wing_autopilot import AutopilotCommand, FixedWingAutopilot
    from uav_sim.vehicles.fixed_wing import FixedWingPreset, create_fixed_wing

    aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8)
    aircraft.reset_trimmed(altitude=120.0)
    pilot = FixedWingAutopilot(aircraft.fw_params)

    dt = 0.02
    steps = 4000
    states = np.zeros((steps, 12))
    telemetry = np.zeros((steps, 4))  # altitude, airspeed, course, bank

    for index in range(steps):
        t = index * dt
        command = AutopilotCommand(
            altitude=120.0 if t < 12 else 175.0,
            airspeed=18.0 if t < 40 else 22.0,
            course=0.0 if t < 25 else np.radians(115.0),
        )
        states[index] = aircraft.state
        velocity = aircraft.velocity
        telemetry[index] = [
            aircraft.state[2],
            aircraft.airspeed,
            np.degrees(np.arctan2(velocity[1], velocity[0])),
            np.degrees(aircraft.state[3]),
        ]
        aircraft.step(pilot.compute(aircraft.state, command, dt), dt)

    return states, telemetry


def simulate_vtol():
    """Hover, transition to wing-borne cruise, and back."""
    from uav_sim.control.vtol_controller import VTOLCommand, VTOLController
    from uav_sim.vehicles.vtol import Tiltrotor

    vtol = Tiltrotor()
    state = np.zeros(12)
    state[2] = 25.0
    vtol.reset(state=state)
    pilot = VTOLController(vtol.vtol_params)
    command = VTOLCommand(altitude=25.0, cruise=False, cruise_airspeed=24.0)

    dt = 0.02
    steps = 6500
    states = np.zeros((steps, 12))
    telemetry = np.zeros((steps, 3))  # tilt, lift fraction, airspeed

    for index in range(steps):
        t = index * dt
        command.cruise = 8.0 <= t < 95.0
        states[index] = vtol.state
        telemetry[index] = [np.degrees(vtol.tilt), vtol.lift_fraction, vtol.airspeed]
        vtol.step(pilot.compute(vtol.state, vtol.tilt, command, dt), dt)

    return states, telemetry


def simulate_swarm():
    """Reynolds flocking."""
    from uav_sim.swarm.reynolds_flocking import ReynoldsFlocking

    rng = np.random.default_rng(7)
    count = 14
    positions = rng.uniform(-10, 10, size=(count, 3))
    positions[:, 2] = rng.uniform(5, 11, size=count)
    velocities = rng.normal(0, 0.8, size=(count, 3))

    # w_mig is Reynolds' migratory urge. Separation, alignment and cohesion
    # all go to zero once the flock is in formation, so without it the flock
    # arranges itself correctly and then mills in place.
    flock = ReynoldsFlocking(r_percept=9.0, r_sep=2.5, w_mig=1.2, world_size=40.0)
    dt = 0.05
    steps = 900
    history = np.zeros((steps, count, 3))

    for index in range(steps):
        history[index] = positions
        heading = 0.22 * index * dt
        migration = 3.0 * np.array([np.cos(heading), np.sin(heading), 0.0])
        forces = flock.compute_forces(positions, velocities, migration_velocity=migration)
        velocities = velocities + forces * dt
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocities = np.where(
            speeds > 4.0, velocities * 4.0 / np.maximum(speeds, 1e-9), velocities
        )
        positions = positions + velocities * dt
        positions[:, 2] = np.clip(positions[:, 2], 3.0, 14.0)

    return history


def simulate_planner():
    """A* through a field of buildings."""
    from uav_sim.environment import default_world
    from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles

    _, obstacles = default_world(world_size=30.0, n_buildings=7, seed=11)
    path = plan_through_obstacles(
        obstacles,
        start=np.array([2.0, 2.0, 4.0]),
        goal=np.array([27.0, 27.0, 10.0]),
        world_size=30,
    )
    return obstacles, np.zeros((0, 3)) if path is None else np.asarray(path)


def simulate_estimation():
    """GPS + IMU fusion against each input on its own.

    The point of the scene is the separation between the three curves:
    dead-reckoning diverging without bound, raw fixes bounded but noisy,
    and the filter below both. That only happens when Q is built from a
    noise density rather than typed in as a dt-independent diagonal.
    """
    from uav_sim.estimation.ekf import ExtendedKalmanFilter
    from uav_sim.estimation.process_noise import constant_acceleration_input_q
    from uav_sim.simulations.common import CRUISE_ALT
    from uav_sim.simulations.standards import (
        SimulationStandard,
        deterministic_truth_trajectory,
    )

    gps_std, imu_std = 0.5, 0.10
    imu_bias = np.array([0.035, -0.025, 0.02])
    standard = SimulationStandard.estimation_benchmark()
    truth, times = deterministic_truth_trajectory(standard, alt=CRUISE_ALT, rx=8.0, ry=6.0)
    dt = float(times[1] - times[0])
    steps = len(truth)
    rng = np.random.default_rng(42)

    def transition(x, u, step):
        return np.array(
            [
                x[0] + x[3] * step,
                x[1] + x[4] * step,
                x[2] + x[5] * step,
                x[3] + u[0] * step,
                x[4] + u[1] * step,
                x[5] + u[2] * step,
            ]
        )

    def jacobian(_x, _u, step):
        matrix = np.eye(6)
        matrix[0, 3] = matrix[1, 4] = matrix[2, 5] = step
        return matrix

    def observation(x):
        return x[:3]

    def observation_jacobian(_x):
        matrix = np.zeros((3, 6))
        matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = 1.0
        return matrix

    ekf = ExtendedKalmanFilter(6, 3, transition, observation, jacobian, observation_jacobian)
    ekf.Q = constant_acceleration_input_q(
        dt, sigma_a=imu_std, sigma_bias=float(np.linalg.norm(imu_bias))
    )
    ekf.R = np.diag([gps_std**2] * 3)
    ekf.x = np.concatenate([truth[0, :3], truth[0, 6:9]])
    ekf.P = np.eye(6) * 0.5

    gps_period = max(1, int(round(1.0 / (5.0 * dt))))
    dead_reckoned = truth[0, :3].copy()
    dead_velocity = truth[0, 6:9].copy()
    previous = truth[0, 6:9].copy()

    fused = np.zeros((steps, 3))
    imu_only = np.zeros((steps, 3))
    fixes = []
    error = np.zeros((steps, 3))  # fused, imu-only, held gps

    held_gps = truth[0, :3].copy()
    for index in range(steps):
        state = truth[index]
        acceleration = (state[6:9] - previous) / dt if index else np.zeros(3)
        previous = state[6:9].copy()
        measured = acceleration + imu_bias + rng.normal(0, imu_std, 3)
        if index:
            dead_velocity = dead_velocity + measured * dt
            dead_reckoned = dead_reckoned + dead_velocity * dt

        ekf.predict(measured, dt)
        if index % gps_period == 0:
            held_gps = state[:3] + rng.normal(0, gps_std, 3)
            ekf.update(held_gps)
            fixes.append(held_gps.copy())

        fused[index] = ekf.x[:3]
        imu_only[index] = dead_reckoned
        error[index] = [
            np.linalg.norm(ekf.x[:3] - state[:3]),
            np.linalg.norm(dead_reckoned - state[:3]),
            np.linalg.norm(held_gps - state[:3]),
        ]

    return truth[:, :3], fused, imu_only, np.array(fixes), error, times


def simulate_perception():
    """Occupancy grid built from 2-D lidar while flying a lawnmower sweep."""
    from uav_sim.environment import default_world
    from uav_sim.path_tracking.pid_controller import CascadedPIDController
    from uav_sim.sensors.lidar import Lidar2D
    from uav_sim.simulations.mission_runner import run_standard_mission
    from uav_sim.simulations.standards import SimulationStandard
    from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

    world, obstacles = default_world(world_size=30.0, n_buildings=6, seed=42)

    lanes = []
    for index, y_lane in enumerate(np.linspace(4.0, 26.0, 5)):
        xs = [3.0, 27.0] if index % 2 == 0 else [27.0, 3.0]
        lanes.extend([[x, y_lane, 10.0] for x in xs])
    sweep = np.array(lanes)

    quad = Quadrotor()
    quad.reset(position=np.array([sweep[0, 0], sweep[0, 1], 0.0]))
    mission = run_standard_mission(
        quad,
        CascadedPIDController(),
        sweep,
        standard=SimulationStandard.flight_coupled(),
        obstacles=obstacles,
    )

    resolution = 0.5
    cells = int(30.0 / resolution)
    log_odds = np.zeros((cells, cells))
    lidar = Lidar2D(num_beams=90, max_range=12.0, noise_std=0.05, seed=42)

    states = mission.states
    scan_at = np.linspace(0, len(states) - 1, 160).astype(int)
    frames = []
    for step in scan_at:
        state = states[step]
        ranges = lidar.sense(state, world)
        yaw = float(state[5])
        for beam, angle in enumerate(lidar.angles):
            distance = ranges[beam]
            direction = np.array([np.cos(yaw + angle), np.sin(yaw + angle)])
            span = int(distance / resolution)
            for tick in range(span + 1):
                point = state[:2] + direction * (tick * resolution)
                col, row = int(point[0] / resolution), int(point[1] / resolution)
                if not (0 <= col < cells and 0 <= row < cells):
                    continue
                if tick == span and distance < lidar.max_range - 0.5:
                    log_odds[col, row] += 0.9
                else:
                    log_odds[col, row] -= 0.4
        np.clip(log_odds, -8.0, 8.0, out=log_odds)
        frames.append(1.0 / (1.0 + np.exp(-log_odds.T)))

    return states[:, :3], np.array(frames), scan_at, obstacles


def simulate_trajectory():
    """Minimum-snap polynomial through a handful of waypoints."""
    from uav_sim.trajectory_planning.min_snap import MinSnapTrajectory

    waypoints = np.array(
        [
            [0.0, 0.0, 2.0],
            [6.0, 4.0, 5.0],
            [12.0, -3.0, 3.5],
            [18.0, 3.0, 6.0],
            [24.0, 0.0, 3.0],
        ]
    )
    segments = np.array(
        [np.linalg.norm(waypoints[i + 1] - waypoints[i]) for i in range(len(waypoints) - 1)]
    )
    durations = np.clip(segments / 3.0, 1.5, 5.0)

    planner = MinSnapTrajectory()
    coefficients = planner.generate(waypoints, durations)
    _, samples = planner.evaluate(coefficients, durations, dt=0.02)
    samples = np.asarray(samples)

    step = 0.02
    velocity = np.gradient(samples, step, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return waypoints, samples, speed


# ══════════════════════════════════════════════════════════════════════════
# Scene painters
# ══════════════════════════════════════════════════════════════════════════


def make_title_scene():
    def draw(figure, progress):
        axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axes.set_facecolor(INK)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.axis("off")

        # Faint engineering grid, matching the site's backdrop.
        for value in np.arange(0, 1.0001, 0.0417):
            axes.axvline(value, color=GRID, lw=0.5, alpha=0.35)
            axes.axhline(value, color=GRID, lw=0.5, alpha=0.35)

        # Attitude indicator, banking gently.
        centre = (0.5, 0.575)
        radius = 0.145
        bank = np.radians(18.0 * np.sin(progress * 2.4 * np.pi))

        # The axes span the whole 16:9 figure, so a true circle in data
        # coordinates would render as a wide ellipse. Squashing x by the
        # aspect ratio makes it read as round.
        aspect = HEIGHT / WIDTH
        theta = np.linspace(0, 2 * np.pi, 200)
        axes.plot(
            centre[0] + radius * np.cos(theta) * aspect,
            centre[1] + radius * np.sin(theta),
            color=SKY,
            lw=2.5,
            alpha=0.95,
        )

        # Sky and ground, clipped to the bezel. Without the clip the fills
        # extend to their bounding box and the dial reads as a square.
        bezel = matplotlib.patches.Ellipse(
            centre, 2 * radius * aspect, 2 * radius, transform=axes.transData
        )
        span = np.linspace(-1.6, 1.6, 60)
        for sign, colour in ((1, SKY), (-1, "#334155")):
            xs = centre[0] + span * radius * aspect * np.cos(bank)
            ys = centre[1] + span * radius * np.sin(bank)
            band = axes.fill_between(
                xs, ys, centre[1] + sign * radius * 1.6, color=colour, alpha=0.28
            )
            band.set_clip_path(bezel)

        span = np.linspace(-1, 1, 60)
        axes.plot(
            centre[0] + span * radius * 0.5625 * np.cos(bank),
            centre[1] + span * radius * np.sin(bank),
            color=TEXT,
            lw=2.0,
        )
        # Fixed aircraft symbol.
        axes.plot([centre[0] - 0.055, centre[0] - 0.018], [centre[1]] * 2, color=AMBER, lw=3)
        axes.plot([centre[0] + 0.018, centre[0] + 0.055], [centre[1]] * 2, color=AMBER, lw=3)
        axes.plot(
            [centre[0] - 0.012, centre[0], centre[0] + 0.012],
            [centre[1], centre[1] - 0.018, centre[1]],
            color=AMBER,
            lw=3,
        )

        title = axes.text(
            0.5,
            0.335,
            "FLYBOTS",
            color=TEXT,
            fontsize=62,
            fontweight="bold",
            ha="center",
            va="center",
            family="DejaVu Sans",
        )
        title.set_path_effects([path_effects.withStroke(linewidth=6, foreground=INK)])
        axes.text(
            0.5,
            0.262,
            "flight algorithms, from scratch",
            color=SKY,
            fontsize=22,
            ha="center",
            va="center",
            family="DejaVu Sans Mono",
        )
        axes.text(
            0.5,
            0.175,
            "multirotor   ·   fixed-wing   ·   VTOL   ·   planning   ·   trajectories\n"
            "estimation   ·   perception   ·   swarms   ·   reinforcement learning",
            color=MUTED,
            fontsize=14,
            linespacing=1.8,
            ha="center",
            va="center",
            family="DejaVu Sans",
        )

    return Scene("", "", 4.0, draw)


def make_quadrotor_scene(states, reference):
    def draw(figure, progress):
        axes = figure.add_axes(FULL_3D_RECT, projection="3d")
        _style_3d(
            axes,
            [(-6.5, 6.5), (-5.5, 5.5), (0, 6)],
            azim=-60 + 28 * progress,
            zoom=FULL_3D_ZOOM,
        )

        cursor = max(2, int(progress * len(states)))
        trail = states[:cursor, :3]
        axes.plot(
            reference[:, 0],
            reference[:, 1],
            reference[:, 2],
            color=GRID,
            lw=1.2,
            ls="--",
            label="reference",
        )
        axes.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=SKY, lw=2.0, label="flown")
        axes.scatter(*trail[-1], color=AMBER, s=70, depthshade=False)
        legend = axes.legend(loc="upper left", fontsize=9, facecolor=PANEL, edgecolor=GRID)
        for text in legend.get_texts():
            text.set_color(TEXT)

        error = np.linalg.norm(states[:cursor, :3] - reference[:cursor], axis=1).mean()
        scene.stats = [(f"{error:.2f} m", "mean error"), ("6", "DOF")]

    scene = Scene(
        "Multirotor",
        "geometric SO(3) tracking · 6DOF rigid body · motor dynamics",
        6.5,
        draw,
        caption="Attitude control on the rotation group, with first-order "
        "motor lag inside the loop.",
    )
    return scene


def make_fixed_wing_scene(states, telemetry):
    def draw(figure, progress):
        axes = figure.add_axes([0.055, 0.30, 0.55, 0.53], projection="3d")
        cursor = max(2, int(progress * len(states)))
        trail = states[:cursor, :3]
        _style_3d(
            axes,
            [
                (states[:, 0].min() - 40, states[:, 0].max() + 40),
                (states[:, 1].min() - 40, states[:, 1].max() + 40),
                (100, 200),
            ],
            elev=24,
            azim=-64 + 24 * progress,
        )
        axes.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=TEAL, lw=2.2)
        axes.scatter(*trail[-1], color=AMBER, s=70, depthshade=False)

        # Telemetry strip.
        times = np.arange(len(telemetry)) * 0.02
        panels = [
            ("altitude [m]", telemetry[:, 0], SKY, (110, 190)),
            ("airspeed [m/s]", telemetry[:, 1], TEAL, (14, 26)),
            ("course [deg]", telemetry[:, 2], VIOLET, (-30, 150)),
        ]
        for index, (label, series, colour, ylim) in enumerate(panels):
            panel = figure.add_axes([0.655, 0.665 - index * 0.19, 0.30, 0.135])
            panel.set_facecolor(PANEL)
            panel.plot(times[:cursor], series[:cursor], color=colour, lw=1.8)
            panel.set_xlim(0, times[-1])
            panel.set_ylim(*ylim)
            panel.tick_params(colors=MUTED, labelsize=7)
            panel.grid(True, color=GRID, alpha=0.5, lw=0.5)
            for spine in panel.spines.values():
                spine.set_color(GRID)
            panel.set_title(label, color=MUTED, fontsize=9, loc="left", pad=6)

        scene.stats = [
            (f"{telemetry[cursor - 1, 1]:.1f}", "m/s"),
            (f"{telemetry[cursor - 1, 0]:.0f}", "m alt"),
        ]

    scene = Scene(
        "Fixed-wing",
        "Beard & McLain aerodynamics · derived-gain autopilot",
        8.0,
        draw,
        caption="Every stability derivative live: static stability, damping, "
        "sideslip, stall. Trim solved numerically.",
    )
    return scene


def make_vtol_scene(states, telemetry):
    def draw(figure, progress):
        cursor = max(2, int(progress * len(states)))
        axes = figure.add_axes([0.055, 0.30, 0.55, 0.53], projection="3d")
        trail = states[:cursor, :3]
        _style_3d(
            axes,
            [
                (states[:, 0].min() - 30, states[:, 0].max() + 30),
                (-40, 40),
                (0, 45),
            ],
            elev=18,
            azim=-72 + 20 * progress,
        )
        axes.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=VIOLET, lw=2.2)
        axes.scatter(*trail[-1], color=AMBER, s=70, depthshade=False)

        times = np.arange(len(telemetry)) * 0.02
        panels = [
            ("rotor tilt [deg]", telemetry[:, 0], AMBER, (-5, 95)),
            ("wing lift share", telemetry[:, 1] * 100, TEAL, (-5, 120)),
            ("airspeed [m/s]", telemetry[:, 2], SKY, (-1, 30)),
        ]
        for index, (label, series, colour, ylim) in enumerate(panels):
            panel = figure.add_axes([0.655, 0.665 - index * 0.19, 0.30, 0.135])
            panel.set_facecolor(PANEL)
            panel.plot(times[:cursor], series[:cursor], color=colour, lw=1.8)
            panel.set_xlim(0, times[-1])
            panel.set_ylim(*ylim)
            panel.tick_params(colors=MUTED, labelsize=7)
            panel.grid(True, color=GRID, alpha=0.5, lw=0.5)
            for spine in panel.spines.values():
                spine.set_color(GRID)
            panel.set_title(label, color=MUTED, fontsize=9, loc="left", pad=6)

        scene.stats = [
            (f"{telemetry[cursor - 1, 0]:.0f}", "deg tilt"),
            (f"{telemetry[cursor - 1, 1] * 100:.0f}%", "wing-borne"),
        ]

    scene = Scene(
        "VTOL transition",
        "hover → cruise → hover · lift handed to the wing",
        8.0,
        draw,
        caption="Altitude held throughout while control authority migrates "
        "from the rotors to the wing.",
    )
    return scene


def make_swarm_scene(history):
    def draw(figure, progress):
        cursor = max(2, int(progress * len(history)))
        tail = max(0, cursor - 220)

        # The migratory urge carries the flock tens of metres, while the
        # flock itself is a couple of metres across. A box big enough for
        # the whole journey renders the agents as specks, so the view
        # tracks the centroid instead and the trails carry the travel.
        centre = history[cursor - 1].mean(axis=0)
        half = 11.0
        axes = figure.add_axes(FULL_3D_RECT, projection="3d")
        _style_3d(
            axes,
            [
                (centre[0] - half, centre[0] + half),
                (centre[1] - half, centre[1] + half),
                (0, 16),
            ],
            elev=26,
            azim=-55 + 40 * progress,
            zoom=FULL_3D_ZOOM,
        )
        for agent in range(history.shape[1]):
            colour = TRACE_COLOURS[agent % len(TRACE_COLOURS)]
            trail = history[tail:cursor, agent]
            axes.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=colour, lw=1.3, alpha=0.75)
            axes.scatter(*trail[-1], color=colour, s=44, depthshade=False)

        travelled = np.linalg.norm(history[cursor - 1].mean(axis=0) - history[0].mean(axis=0))
        scene.stats = [(str(history.shape[1]), "agents"), (f"{travelled:.0f} m", "travelled")]

    scene = Scene(
        "Swarms",
        "Reynolds flocking · separation, alignment, cohesion",
        5.5,
        draw,
        caption="Also: consensus formation, virtual structures, "
        "leader-follower and Voronoi coverage.",
    )
    return scene


def make_planner_scene(obstacles, path):
    def draw(figure, progress):
        axes = figure.add_axes(FULL_3D_RECT, projection="3d")
        _style_3d(
            axes,
            [(0, 30), (0, 30), (0, 16)],
            elev=32,
            azim=-58 + 30 * progress,
            zoom=FULL_3D_ZOOM,
        )

        for obstacle in obstacles:
            low = np.asarray(obstacle.min_corner, dtype=float)
            high = np.asarray(obstacle.max_corner, dtype=float)
            xs = np.array([low[0], high[0], high[0], low[0], low[0]])
            ys = np.array([low[1], low[1], high[1], high[1], low[1]])
            # Wireframes in the grid colour disappear against the grid.
            for level in (low[2], high[2]):
                axes.plot(xs, ys, level, color=MUTED, lw=1.1, alpha=0.55)
            for corner in range(4):
                axes.plot(
                    [xs[corner]] * 2,
                    [ys[corner]] * 2,
                    [low[2], high[2]],
                    color=MUTED,
                    lw=1.1,
                    alpha=0.55,
                )

        if len(path):
            cursor = max(2, int(progress * len(path)))
            axes.plot(path[:, 0], path[:, 1], path[:, 2], color=GRID, lw=1.0, ls=":")
            axes.plot(path[:cursor, 0], path[:cursor, 1], path[:cursor, 2], color=AMBER, lw=2.6)
            axes.scatter(*path[cursor - 1], color=SKY, s=70, depthshade=False)
        scene.stats = [(str(len(path)), "waypoints"), ("3D", "A*")]

    scene = Scene(
        "Planning",
        "A* · RRT* · PRM · potential fields · coverage",
        5.5,
        draw,
        caption="Volumetric search through a city of obstacles, with costmap "
        "layers for inflation and social cost.",
    )
    return scene


def make_estimation_scene(truth, fused, imu_only, fixes, error, times):
    def draw(figure, progress):
        cursor = max(2, int(progress * len(truth)))

        axes = figure.add_axes([0.055, 0.20, 0.45, 0.62], projection="3d")
        _style_3d(axes, [(5, 25), (7, 23), (6, 18)], elev=26, azim=-62 + 28 * progress)
        axes.plot(
            truth[:cursor, 0],
            truth[:cursor, 1],
            truth[:cursor, 2],
            color=TEXT,
            lw=2.0,
            alpha=0.85,
        )
        shown = max(1, int(progress * len(fixes)))
        axes.scatter(
            fixes[:shown, 0],
            fixes[:shown, 1],
            fixes[:shown, 2],
            color=TEAL,
            s=9,
            alpha=0.35,
            depthshade=False,
        )
        axes.plot(
            fused[:cursor, 0],
            fused[:cursor, 1],
            fused[:cursor, 2],
            color=SKY,
            lw=1.8,
        )
        drift = min(cursor, len(imu_only))
        axes.plot(
            imu_only[:drift, 0],
            imu_only[:drift, 1],
            imu_only[:drift, 2],
            color=ROSE,
            lw=1.4,
            alpha=0.8,
            ls="--",
        )
        axes.set_title("truth · GPS fixes · fused · IMU-only", color=TEXT, fontsize=11, pad=2)

        panel = figure.add_axes([0.56, 0.24, 0.39, 0.54])
        panel.set_facecolor(PANEL)
        span = times[:cursor]
        panel.plot(span, error[:cursor, 1], color=ROSE, lw=2.0, label="IMU only")
        panel.plot(span, error[:cursor, 2], color=TEAL, lw=1.0, alpha=0.6, label="raw GPS")
        panel.plot(span, error[:cursor, 0], color=SKY, lw=2.2, label="EKF")
        panel.set_xlim(0, times[-1])
        panel.set_ylim(0, max(4.0, float(error[:, 1].max()) * 1.05))
        panel.set_xlabel("time [s]", color=MUTED, fontsize=10)
        panel.set_ylabel("position error [m]", color=MUTED, fontsize=10)
        panel.tick_params(colors=MUTED, labelsize=8)
        panel.grid(True, color=GRID, alpha=0.5, lw=0.5)
        for spine in panel.spines.values():
            spine.set_color(GRID)
        legend = panel.legend(loc="upper left", fontsize=9, facecolor=PANEL, edgecolor=GRID)
        for text in legend.get_texts():
            text.set_color(MUTED)
        panel.set_title(
            "the filter has to beat its own sensors", color=TEXT, fontsize=12, loc="left", pad=8
        )

        scene.stats = [
            (f"{error[:cursor, 0].mean():.2f} m", "EKF"),
            (f"{error[:cursor, 1].mean():.1f} m", "dead reckoning"),
        ]

    scene = Scene(
        "Estimation",
        "EKF · UKF · particle filter · complementary filter",
        6.5,
        draw,
        caption="Dead reckoning diverges, GPS is noisy, the fusion sits under "
        "both — which only holds when Q is scaled by the timestep.",
    )
    return scene


def make_perception_scene(track, grids, scan_at, obstacles):
    def draw(figure, progress):
        index = min(len(grids) - 1, int(progress * len(grids)))
        cursor = int(scan_at[index])

        axes = figure.add_axes([0.055, 0.20, 0.42, 0.62], projection="3d")
        _style_3d(axes, [(0, 30), (0, 30), (0, 20)], elev=30, azim=-64 + 24 * progress)
        for obstacle in obstacles:
            low, high = obstacle.min_corner, obstacle.max_corner
            xs = [low[0], high[0], high[0], low[0], low[0]]
            ys = [low[1], low[1], high[1], high[1], low[1]]
            axes.plot(xs, ys, [high[2]] * 5, color=GRID, lw=1.1, alpha=0.9)
            for corner in range(4):
                axes.plot(
                    [xs[corner]] * 2,
                    [ys[corner]] * 2,
                    [low[2], high[2]],
                    color=GRID,
                    lw=1.1,
                    alpha=0.9,
                )
        axes.plot(track[:cursor, 0], track[:cursor, 1], track[:cursor, 2], color=AMBER, lw=2.0)
        if cursor:
            axes.scatter(*track[cursor - 1], color=SKY, s=70, depthshade=False)
        axes.set_title("lidar sweep", color=TEXT, fontsize=11, pad=2)

        panel = figure.add_axes([0.53, 0.19, 0.40, 0.64])
        panel.set_facecolor(PANEL)
        panel.imshow(
            grids[index],
            origin="lower",
            extent=[0, 30, 0, 30],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        if cursor:
            panel.plot(track[cursor - 1, 0], track[cursor - 1, 1], "o", color=SKY, ms=7)
        panel.set_xlabel("x [m]", color=MUTED, fontsize=10)
        panel.set_ylabel("y [m]", color=MUTED, fontsize=10)
        panel.tick_params(colors=MUTED, labelsize=8)
        for spine in panel.spines.values():
            spine.set_color(GRID)
        panel.set_title("occupancy grid, log-odds", color=TEXT, fontsize=12, loc="left", pad=8)

        known = float(np.mean(np.abs(grids[index] - 0.5) > 0.02) * 100.0)
        scene.stats = [(f"{known:.0f}%", "mapped"), ("90", "beams")]

    scene = Scene(
        "Perception",
        "occupancy mapping · EKF-SLAM · visual servoing · gimbal tracking",
        6.0,
        draw,
        caption="A map assembled from range returns alone, one Bayesian log-odds update per beam.",
    )
    return scene


def make_trajectory_scene(waypoints, samples, speed):
    def draw(figure, progress):
        cursor = max(2, int(progress * len(samples)))

        axes = figure.add_axes([0.055, 0.20, 0.45, 0.62], projection="3d")
        _style_3d(axes, [(-2, 26), (-6, 6), (0, 8)], elev=24, azim=-66 + 30 * progress)
        axes.plot(samples[:, 0], samples[:, 1], samples[:, 2], color=GRID, lw=1.0, ls=":")
        axes.plot(
            samples[:cursor, 0],
            samples[:cursor, 1],
            samples[:cursor, 2],
            color=VIOLET,
            lw=2.8,
        )
        axes.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            waypoints[:, 2],
            color=AMBER,
            s=70,
            marker="D",
            depthshade=False,
        )
        axes.scatter(*samples[cursor - 1], color=SKY, s=80, depthshade=False)
        axes.set_title("minimum-snap through waypoints", color=TEXT, fontsize=11, pad=2)

        panel = figure.add_axes([0.56, 0.24, 0.39, 0.54])
        panel.set_facecolor(PANEL)
        stamps = np.arange(len(speed)) * 0.02
        panel.plot(stamps[:cursor], speed[:cursor], color=TEAL, lw=2.4)
        panel.fill_between(stamps[:cursor], speed[:cursor], color=TEAL, alpha=0.14)
        panel.set_xlim(0, stamps[-1])
        panel.set_ylim(0, float(speed.max()) * 1.15)
        panel.set_xlabel("time [s]", color=MUTED, fontsize=10)
        panel.set_ylabel("speed [m/s]", color=MUTED, fontsize=10)
        panel.tick_params(colors=MUTED, labelsize=8)
        panel.grid(True, color=GRID, alpha=0.5, lw=0.5)
        for spine in panel.spines.values():
            spine.set_color(GRID)
        panel.set_title(
            "continuous through every knot", color=TEXT, fontsize=12, loc="left", pad=8
        )

        scene.stats = [(str(len(waypoints)), "waypoints"), ("7th", "order")]

    scene = Scene(
        "Trajectory planning",
        "min-snap · quintic · polynomial · Frenet",
        5.5,
        draw,
        caption="Position, velocity, acceleration and jerk all continuous "
        "across the joins — the derivative a quadrotor actually feels.",
    )
    return scene


def make_learning_scene(curve, trajectories):
    def draw(figure, progress):
        cursor = max(2, int(progress * len(curve)))

        panel = figure.add_axes([0.075, 0.24, 0.40, 0.55])
        panel.set_facecolor(PANEL)
        panel.plot(np.arange(cursor), curve[:cursor], color=SKY, lw=2.4)
        panel.fill_between(np.arange(cursor), curve[:cursor], color=SKY, alpha=0.14)
        panel.set_xlim(0, len(curve))
        panel.set_ylim(min(0, curve.min() * 1.1), curve.max() * 1.15)
        panel.set_xlabel("iteration", color=MUTED, fontsize=10)
        panel.set_ylabel("episode return", color=MUTED, fontsize=10)
        panel.tick_params(colors=MUTED, labelsize=8)
        panel.grid(True, color=GRID, alpha=0.5, lw=0.5)
        for spine in panel.spines.values():
            spine.set_color(GRID)
        panel.set_title("learning curve", color=TEXT, fontsize=12, loc="left", pad=8)

        axes = figure.add_axes([0.52, 0.20, 0.44, 0.62], projection="3d")
        _style_3d(axes, [(-3, 3), (-3, 3), (0, 6)], elev=20, azim=-60 + 35 * progress)
        axes.scatter(0, 0, 3.0, color=AMBER, s=110, marker="*", depthshade=False)
        span = max(2, int(progress * max(len(t) for t in trajectories)))
        for index, trajectory in enumerate(trajectories):
            visible = trajectory[: min(span, len(trajectory))]
            if len(visible) < 2:
                continue
            colour = TRACE_COLOURS[index % len(TRACE_COLOURS)]
            axes.plot(visible[:, 0], visible[:, 1], visible[:, 2], color=colour, lw=1.6)
        axes.set_title("learned hover", color=TEXT, fontsize=12, pad=2)

        scene.stats = [(f"{curve[cursor - 1]:.0f}", "return"), ("76", "parameters")]

    scene = Scene(
        "Reinforcement learning",
        "6 environments · pure-NumPy trainer · no deep-learning stack",
        7.0,
        draw,
        caption="A linear policy, trained by random search, learns to hold "
        "position from a randomised upset.",
    )
    return scene


ATLAS_COLUMNS = 3
ATLAS_ROWS = 2
ATLAS_TILE = (550, 275)
ATLAS_GAP = 28
ATLAS_LABEL_HEIGHT = 46
ATLAS_FRAMES_PER_TILE = 10
ATLAS_SECONDS_PER_PAGE = 1.5


def load_atlas_tiles():
    """One looping thumbnail per simulation, read from its rendered GIF.

    The scene this feeds is the only place in the video that claims to show
    *every* algorithm, so it is driven by the same catalogue the CLI walks
    rather than by a list maintained here — a simulation added tomorrow
    appears in the montage without anyone remembering to add it.
    """
    from PIL import Image, ImageSequence

    from uav_sim.cli.catalogue import discover

    width, height = ATLAS_TILE
    tiles = []
    for entry in discover():
        gif = entry.gif
        if gif is None:
            continue
        with Image.open(gif) as source:
            total = getattr(source, "n_frames", 1)
            # Start a fifth of the way in. A simulation's opening frames are
            # an empty axis and a trace that has not been drawn yet, which
            # is the one thing a thumbnail must not show.
            first = int(0.2 * (total - 1))
            span = max(1, total - 1 - first)
            wanted = {
                first + int(round(index * span / max(1, ATLAS_FRAMES_PER_TILE - 1)))
                for index in range(ATLAS_FRAMES_PER_TILE)
            }
            picked = {}
            # One forward pass: GIF frames decode sequentially anyway, so
            # seeking back and forth would re-decode from the start.
            for index, frame in enumerate(ImageSequence.Iterator(source)):
                if index in wanted:
                    picked[index] = np.asarray(
                        frame.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                if len(picked) == len(wanted):
                    break
        if not picked:
            continue
        tiles.append(
            (
                entry.name.replace("_", " "),
                entry.category.replace("_", " "),
                np.stack([picked[key] for key in sorted(picked)]),
            )
        )
    return tiles


def make_atlas_scene(tiles):
    """Every simulation in the library, six at a time.

    Forty-two tiles on screen at once is a contact sheet nobody can read.
    Paging through them keeps each one large enough to recognise what the
    algorithm is doing, which is the only reason to show it at all.
    """
    width, height = ATLAS_TILE
    per_page = ATLAS_COLUMNS * ATLAS_ROWS
    pages = int(np.ceil(len(tiles) / per_page))
    cell_h = height + ATLAS_LABEL_HEIGHT
    grid_w = ATLAS_COLUMNS * width + (ATLAS_COLUMNS - 1) * ATLAS_GAP
    grid_h = ATLAS_ROWS * cell_h + (ATLAS_ROWS - 1) * ATLAS_GAP
    origin_x = (WIDTH - grid_w) // 2
    origin_y = 205

    ink = np.array([int(INK[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.uint8)

    def draw(figure, progress):
        position = min(progress, 0.9999) * pages
        page = int(position)
        # Dip to the background between pages: a hard cut at this size
        # reads as a glitch, a dissolve as a smear.
        within = position - page
        alpha = float(np.clip(min(within, 1.0 - within) / 0.16, 0.0, 1.0))

        montage = np.tile(ink, (grid_h, grid_w, 1)).astype(np.float32)

        shown = tiles[page * per_page : (page + 1) * per_page]
        for index, (_, _, frames) in enumerate(shown):
            row, column = divmod(index, ATLAS_COLUMNS)
            phase = int(progress * 12.0 * len(frames) + index * 3) % len(frames)
            top = row * (cell_h + ATLAS_GAP)
            left = column * (width + ATLAS_GAP)
            patch = montage[top : top + height, left : left + width]
            patch *= 1.0 - alpha
            patch += alpha * frames[phase]

        figure.figimage(
            np.clip(montage, 0, 255).astype(np.uint8),
            xo=origin_x,
            yo=HEIGHT - (origin_y + grid_h),
            zorder=0,
        )

        for index, (label, category, _) in enumerate(shown):
            row, column = divmod(index, ATLAS_COLUMNS)
            centre_x = origin_x + column * (width + ATLAS_GAP) + width / 2
            baseline = origin_y + row * (cell_h + ATLAS_GAP) + height + 14
            figure.text(
                centre_x / WIDTH,
                1.0 - baseline / HEIGHT,
                label,
                color=TEXT,
                fontsize=14,
                alpha=alpha,
                ha="center",
                va="top",
                family="DejaVu Sans",
            )
            figure.text(
                centre_x / WIDTH,
                1.0 - (baseline + 24) / HEIGHT,
                category,
                color=MUTED,
                fontsize=10,
                alpha=alpha,
                ha="center",
                va="top",
                family="DejaVu Sans Mono",
            )

        scene.stats = [(str(len(tiles)), "simulations"), (str(categories), "domains")]

    categories = len({category for _, category, _ in tiles})
    scene = Scene(
        "The whole atlas",
        "every simulation in the library, as it renders",
        pages * ATLAS_SECONDS_PER_PAGE,
        draw,
        caption="Each tile is that simulation's own animation — the GIF the "
        "repository ships, not a mock-up of one.",
    )
    return scene


def make_outro_scene():
    def draw(figure, progress):
        axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axes.set_facecolor(INK)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.axis("off")
        for value in np.arange(0, 1.0001, 0.0417):
            axes.axvline(value, color=GRID, lw=0.5, alpha=0.3)
            axes.axhline(value, color=GRID, lw=0.5, alpha=0.3)

        axes.text(
            0.5,
            0.635,
            "pip install flybots",
            color=TEXT,
            fontsize=52,
            fontweight="bold",
            ha="center",
            va="center",
            family="DejaVu Sans Mono",
        )

        commands = [
            ("flybots list", "browse 42 simulations"),
            ("flybots run pid_hover", "render one to a GIF"),
            ("flybots train hover", "teach a drone to fly"),
        ]
        reveal = int(np.clip(progress * 4.0, 0, len(commands)))
        for index, (command, description) in enumerate(commands[:reveal]):
            y = 0.475 - index * 0.075
            axes.text(
                0.36,
                y,
                command,
                color=SKY,
                fontsize=21,
                ha="left",
                va="center",
                family="DejaVu Sans Mono",
            )
            axes.text(
                0.615,
                y,
                description,
                color=MUTED,
                fontsize=15,
                ha="left",
                va="center",
                family="DejaVu Sans",
            )

        axes.text(
            0.5,
            0.175,
            "github.com/guilyx/flybots",
            color=MUTED,
            fontsize=17,
            ha="center",
            va="center",
            family="DejaVu Sans Mono",
        )
        axes.text(
            0.5,
            0.115,
            "MIT licensed",
            color=GRID,
            fontsize=13,
            ha="center",
            va="center",
            family="DejaVu Sans",
        )

    return Scene("", "", 5.0, draw)


# ══════════════════════════════════════════════════════════════════════════
# Encoder
# ══════════════════════════════════════════════════════════════════════════


def render(scenes: list[Scene], output: Path, fps: int = FPS) -> Path:
    """Render every scene and encode to MP4."""
    if not 1 <= fps <= 240:
        raise ValueError(f"fps must be in [1, 240], got {fps}")

    try:
        import imageio_ffmpeg
    except ImportError:  # pragma: no cover - depends on the optional extra
        print(
            "imageio-ffmpeg is required to encode the video.\n  uv pip install imageio-ffmpeg",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    total_frames = sum(int(scene.seconds * fps) for scene in scenes)

    # Argument list, no shell: the binary comes from imageio-ffmpeg, fps is
    # bounds-checked above, and the output path is exactly the file this
    # function's caller asked to be written -- the same trust boundary as
    # `open(output, "wb")`. Static scanners flag any Popen with a non-literal
    # argument regardless of shell=False; this one has none.
    process = subprocess.Popen(  # noqa: S603  # nosec B603  # nosemgrep
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            str(CRF),
            # Web-friendly: move the index to the front so it streams.
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    fade_frames = int(0.35 * fps)
    written = 0

    for scene in scenes:
        frames = int(scene.seconds * fps)
        for frame in range(frames):
            progress = frame / max(frames - 1, 1)
            figure = Figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=INK)
            scene.draw(figure, progress)
            if scene.title:
                _draw_chrome(figure, scene, written / total_frames)

            # Cross-fade at the seams.
            if frame < fade_frames:
                _fade(figure, 1.0 - frame / fade_frames)
            elif frame > frames - fade_frames:
                _fade(figure, (frame - (frames - fade_frames)) / fade_frames)

            # A bare Figure() carries a FigureCanvasBase, which cannot
            # rasterise. Attaching an Agg canvas explicitly is what makes
            # buffer_rgba() available — and keeps us off pyplot's global
            # figure registry, which would otherwise leak a figure per frame.
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            buffer = np.asarray(canvas.buffer_rgba())[:, :, :3]
            process.stdin.write(buffer.tobytes())
            written += 1

            if written % 30 == 0:
                percent = 100.0 * written / total_frames
                print(
                    f"\r  rendering {written}/{total_frames} ({percent:.0f}%)", end="", flush=True
                )

    print()
    process.stdin.close()
    process.wait()
    return output


def write_poster(scenes: list[Scene], output: Path, *, scene_index: int, progress: float) -> Path:
    """Render one frame as the video's poster image.

    Generated from the same scene list as the video so the two cannot drift
    apart — the poster used to be a hand-extracted frame, which is exactly
    the kind of artefact that silently keeps showing last year's behaviour.
    """
    scene = scenes[scene_index]
    figure = Figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=INK)
    scene.draw(figure, progress)
    if scene.title:
        _draw_chrome(figure, scene, progress)

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[:, :, :3]

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with matplotlib
        print("Pillow is required to write the poster.", file=sys.stderr)
        raise SystemExit(1) from None

    # Half size: it is a poster behind a play button, not a still to study.
    Image.fromarray(frame).resize((WIDTH // 2, HEIGHT // 2), Image.Resampling.LANCZOS).save(output)
    return output


def build_scenes(reuse_policy: bool = True) -> list[Scene]:
    print("simulating quadrotor...", flush=True)
    quad_states, quad_reference = simulate_quadrotor()

    print("simulating fixed-wing...", flush=True)
    fw_states, fw_telemetry = simulate_fixed_wing()

    print("simulating VTOL...", flush=True)
    vtol_states, vtol_telemetry = simulate_vtol()

    print("simulating swarm...", flush=True)
    swarm_history = simulate_swarm()

    print("planning...", flush=True)
    obstacles, path = simulate_planner()

    print("planning a min-snap trajectory...", flush=True)
    waypoints, samples, speed = simulate_trajectory()

    print("estimating...", flush=True)
    truth, fused, imu_only, fixes, error, times = simulate_estimation()

    print("mapping...", flush=True)
    map_track, grids, scan_at, map_obstacles = simulate_perception()

    curve, trajectories = _learning_material(reuse_policy)

    print("loading the simulation atlas...", flush=True)
    tiles = load_atlas_tiles()

    scenes = [
        make_title_scene(),
        make_quadrotor_scene(quad_states, quad_reference),
        make_fixed_wing_scene(fw_states, fw_telemetry),
        make_vtol_scene(vtol_states, vtol_telemetry),
        make_planner_scene(obstacles, path),
        make_trajectory_scene(waypoints, samples, speed),
        make_estimation_scene(truth, fused, imu_only, fixes, error, times),
        make_perception_scene(map_track, grids, scan_at, map_obstacles),
        make_swarm_scene(swarm_history),
        make_learning_scene(curve, trajectories),
    ]
    if tiles:
        scenes.append(make_atlas_scene(tiles))
    else:
        # A clone without the LFS payload has the GIF pointers but not the
        # frames. Rendering 42 empty rectangles is worse than not claiming
        # to show the atlas at all.
        print("no simulation GIFs found — skipping the atlas scene", flush=True)
    scenes.append(make_outro_scene())
    return scenes


def _learning_material(reuse_policy: bool):
    """Learning curve and rollout trajectories for the RL scene.

    Training from scratch takes roughly an hour, so a previously saved
    policy is reused unless ``--retrain`` is given. Re-rendering the video
    after a copy tweak should not cost an hour of CPU.
    """
    from uav_sim.gym import make
    from uav_sim.gym.policy import MLPPolicy
    from uav_sim.gym.train import TrainConfig, rollout, train

    policy_path = Path("policies/hover.npz")
    env = make("hover", seed=0)

    curve_path = Path("policies/hover_curve.npy")
    if reuse_policy and policy_path.exists() and curve_path.exists():
        print(f"reusing {policy_path}...", flush=True)
        policy = MLPPolicy.load(policy_path)
        curve = np.load(curve_path)
    else:
        print("training a hover policy (roughly an hour)...", flush=True)
        # These are the settings the training documentation quotes. Cutting
        # episodes_per_candidate to save time also costs a lot of return —
        # with fewer episodes the finite differences stop resolving the
        # difference between candidates, and the run plateaus early.
        result = train(
            "hover",
            config=TrainConfig(
                iterations=60,
                directions=8,
                top_directions=4,
                episodes_per_candidate=8,
                step_size=0.05,
                noise=0.05,
                seed=0,
            ),
        )
        policy = result.policy
        holdout = np.array([row["holdout_return"] for row in result.history])
        # Forward-fill the sparse held-out evaluations into a smooth curve.
        curve = _forward_fill(holdout)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(policy_path)
        np.save(curve_path, curve)

    trajectories = []
    for episode in range(4):
        rollout(env, policy, seed=1000 + episode)
        trajectories.append(env.trajectory)
    env.close()
    return curve, trajectories


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Replace NaNs with the last finite value."""
    filled = np.array(values, dtype=float)
    last = 0.0
    for index, value in enumerate(filled):
        if np.isnan(value):
            filled[index] = last
        else:
            last = value
    return filled


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the promo video.")
    parser.add_argument("--output", default="media/promo.mp4", type=Path)
    parser.add_argument(
        "--poster",
        type=Path,
        default=None,
        help="also write a poster frame here (defaults to <output>-poster.png)",
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help=(
            "train a fresh hover policy instead of reusing policies/hover.npz. "
            "Takes about an hour; the saved policy is reused by default."
        ),
    )
    args = parser.parse_args()

    scenes = build_scenes(reuse_policy=not args.retrain)
    total = sum(scene.seconds for scene in scenes)
    print(f"rendering {len(scenes)} scenes, {total:.0f}s at {args.fps} fps")
    path = render(scenes, args.output, args.fps)
    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path} ({size_mb:.1f} MB, {total:.0f}s)")

    poster = args.poster or path.with_name(f"{path.stem}-poster.png")
    # The multirotor scene, two thirds through: the figure-8 trail is drawn
    # and the airframe is mid-turn.
    written = write_poster(scenes, poster, scene_index=POSTER_SCENE, progress=0.66)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
