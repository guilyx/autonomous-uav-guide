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
WIDTH, HEIGHT = 1920, 1080
DPI = 120


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


def _style_3d(axes, limits, *, elev=22, azim=-60):
    axes.set_facecolor(INK)
    axes.set_xlim(limits[0])
    axes.set_ylim(limits[1])
    axes.set_zlim(limits[2])
    axes.view_init(elev=elev, azim=azim)
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
    quad.reset(position=np.array([0.0, 0.0, 3.0]))
    controller = GeometricController()

    dt = 0.005
    steps = 8000
    states = np.zeros((steps, 12))
    reference = np.zeros((steps, 3))

    amplitude = np.array([5.0, 4.0, 1.0])
    omega = np.array([0.32, 0.64, 0.24])

    for index in range(steps):
        t = index * dt
        phase = omega * t
        target = np.array([0.0, 0.0, 3.0]) + amplitude * np.sin(phase)
        velocity = amplitude * omega * np.cos(phase)
        acceleration = -amplitude * omega**2 * np.sin(phase)

        reference[index] = target
        states[index] = quad.state
        quad.step(controller.compute(quad.state, target, velocity, acceleration), dt)

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

    flock = ReynoldsFlocking(r_percept=9.0, r_sep=2.5, world_size=40.0)
    dt = 0.05
    steps = 900
    history = np.zeros((steps, count, 3))

    for index in range(steps):
        history[index] = positions
        velocities = velocities + flock.compute_forces(positions, velocities) * dt
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
            "AUTONOMOUS UAV",
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
            "multirotor   ·   fixed-wing   ·   VTOL   ·   swarms   ·   reinforcement learning",
            color=MUTED,
            fontsize=14,
            ha="center",
            va="center",
            family="DejaVu Sans",
        )

    return Scene("", "", 4.0, draw)


def make_quadrotor_scene(states, reference):
    def draw(figure, progress):
        axes = figure.add_axes([0.08, 0.16, 0.84, 0.66], projection="3d")
        _style_3d(axes, [(-6.5, 6.5), (-5.5, 5.5), (0, 6)], azim=-60 + 28 * progress)

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
        axes = figure.add_axes([0.08, 0.16, 0.84, 0.66], projection="3d")
        _style_3d(axes, [(-25, 25), (-25, 25), (0, 16)], elev=26, azim=-55 + 40 * progress)
        cursor = max(2, int(progress * len(history)))
        tail = max(0, cursor - 90)
        for agent in range(history.shape[1]):
            colour = TRACE_COLOURS[agent % len(TRACE_COLOURS)]
            trail = history[tail:cursor, agent]
            axes.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=colour, lw=1.3, alpha=0.75)
            axes.scatter(*trail[-1], color=colour, s=34, depthshade=False)
        scene.stats = [(str(history.shape[1]), "agents"), ("3", "rules")]

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
        axes = figure.add_axes([0.08, 0.16, 0.84, 0.66], projection="3d")
        _style_3d(axes, [(0, 30), (0, 30), (0, 16)], elev=32, azim=-58 + 30 * progress)

        for obstacle in obstacles:
            low = np.asarray(obstacle.min_corner, dtype=float)
            high = np.asarray(obstacle.max_corner, dtype=float)
            xs = np.array([low[0], high[0], high[0], low[0], low[0]])
            ys = np.array([low[1], low[1], high[1], high[1], low[1]])
            for level in (low[2], high[2]):
                axes.plot(xs, ys, level, color=GRID, lw=1.1, alpha=0.9)
            for corner in range(4):
                axes.plot(
                    [xs[corner]] * 2,
                    [ys[corner]] * 2,
                    [low[2], high[2]],
                    color=GRID,
                    lw=1.1,
                    alpha=0.9,
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
            "pip install uav-sim",
            color=TEXT,
            fontsize=52,
            fontweight="bold",
            ha="center",
            va="center",
            family="DejaVu Sans Mono",
        )

        commands = [
            ("uav-sim list", "browse 40+ simulations"),
            ("uav-sim run pid_hover", "render one to a GIF"),
            ("uav-sim train hover", "teach a drone to fly"),
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
            "github.com/guilyx/autonomous-uav-guide",
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
            "20",
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

    curve, trajectories = _learning_material(reuse_policy)

    return [
        make_title_scene(),
        make_quadrotor_scene(quad_states, quad_reference),
        make_fixed_wing_scene(fw_states, fw_telemetry),
        make_vtol_scene(vtol_states, vtol_telemetry),
        make_planner_scene(obstacles, path),
        make_swarm_scene(swarm_history),
        make_learning_scene(curve, trajectories),
        make_outro_scene(),
    ]


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


if __name__ == "__main__":
    main()
