# Erwin Lejeune - 2026-08-18
"""Quadrotor, hexacopter and coaxial octocopter flown by one derived mixer.

Three airframes with four, six and eight rotors fly the same closed-loop
box, under the same cascaded controller, with no airframe-specific mixing
code anywhere. Each aircraft's allocation matrix is built from its own
rotor positions and spin directions; the controllers only ever speak in
body wrench, so nothing above the mixer knows how many rotors are below it.

What the run is set up to show:

* **The mixer is the only thing that differs.** The controllers are
  identical bar their gains, which are derived from each airframe's inertia
  (constant angular acceleration per unit rate error) rather than tuned by
  hand per aircraft.
* **Where the thrust goes.** The lower panel plots the hexacopter's six
  rotor thrusts. They split on the roll legs and rejoin on the straights,
  and the split is the ``tau_x = sum(y_i f_i)`` row of the allocation
  matrix showing up as newtons.
* **Yaw is a spin pattern, not an arm length.** The last leg commands a
  90 degree heading change, which the airframe flies purely on the
  reaction torque of its rotors — the six thrusts fan out into the
  alternating ``+ - + - + -`` pattern and the aircraft turns.

The X8's eight rotors sit on four arms, so it has the roll authority of a
quadrotor with twice the thrust and a redundant column for every arm; its
lower rotors run faster than its upper ones throughout, because they work
in the upper rotors' wake.

Reference: M. Achtelik, K.-M. Doth, D. Gurdan, J. Stumpf, "Design of a
Multi Rotor MAV with regard to Efficiency, Dynamics and Redundancy," AIAA
Guidance, Navigation, and Control Conference, 2012. DOI: 10.2514/6.2012-4779
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.control.attitude_controller import AttitudeController, AttitudeControllerConfig
from uav_sim.control.position_controller import PositionController, PositionControllerConfig
from uav_sim.control.rate_controller import RateControllerConfig
from uav_sim.control.velocity_controller import VelocityController, VelocityControllerConfig
from uav_sim.logging import SimLogger
from uav_sim.vehicles.multirotor import Multirotor
from uav_sim.vehicles.presets import VehiclePreset, create_multirotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists, draw_multirotor_3d

matplotlib.use("Agg")

WORLD_SIZE = 40.0
Z_MAX = 18.0
DT = 0.005
LEG_SECONDS = 7.0

# A 90 degree yaw step would be a fine thing to ask a fixed-wing rudder for
# and a bad thing to ask a multirotor for: yaw comes from rotor drag, so
# kappa (16 mm here) is the lever arm, and a step demand saturates every
# rotor before the airframe has turned ten degrees. The command is ramped
# instead, at a rate a real flight stack would use.
YAW_TARGET = np.pi / 2
YAW_RATE = 0.35

# Angular acceleration the rate loop asks for per unit rate error [1/s].
# The library's default gains are 0.055 Nm/(rad/s) on a 0.0082 kg m^2 axis
# and 0.10 on 0.0148; both work out to this, so scaling by inertia leaves
# the 250 mm quadrotor exactly as tuned and gives the heavier airframes the
# same closed-loop bandwidth instead of a third of it.
RATE_ACCEL = 6.7

AIRFRAMES = (
    (VehiclePreset.RACING_250, 8.0, "tab:cyan", "Quadrotor 250 (4 rotors)"),
    (VehiclePreset.HEX_S550, 20.0, "tab:orange", "Hex S550 (6 rotors)"),
    (VehiclePreset.OCTO_X8, 32.0, "tab:green", "Octo X8 (8 coaxial)"),
)

#: Where the hexacopter's rotors sit, for the thrust-panel legend. The
#: layout starts at the rear left and runs counter-clockwise.
HEX_ROTOR_NAMES = ("rear-L", "rear-R", "right", "front-R", "front-L", "left")


def _waypoints(lane: float) -> list[np.ndarray]:
    """A climbing box, then a station-keeping leg for the yaw turn."""
    return [
        np.array([14.0, lane - 5.0, 6.0]),
        np.array([26.0, lane - 5.0, 6.0]),
        np.array([26.0, lane + 5.0, 11.0]),
        np.array([14.0, lane + 5.0, 11.0]),
        np.array([14.0, lane - 5.0, 7.0]),
        np.array([14.0, lane - 5.0, 7.0]),
    ]


def _command(t: float, legs: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Target position and heading at time *t*."""
    leg = min(int(t // LEG_SECONDS), len(legs) - 1)
    if leg < len(legs) - 1:
        return legs[leg], 0.0
    elapsed = t - (len(legs) - 1) * LEG_SECONDS
    return legs[leg], float(min(YAW_RATE * elapsed, YAW_TARGET))


class _Pilot:
    """Position → velocity → attitude → rate, with inertia-derived gains.

    Composed here rather than through :class:`FlightController` so that the
    mass and the inertia of *this* airframe reach the loops that need them.
    """

    def __init__(self, aircraft: Multirotor) -> None:
        params = aircraft.params
        self.position = PositionController(
            PositionControllerConfig(kp=1.0, kd=0.7, max_velocity=3.0)
        )
        self.velocity = VelocityController(
            VelocityControllerConfig(mass=params.mass, gravity=params.gravity, max_tilt=0.30)
        )
        self.attitude = AttitudeController(
            AttitudeControllerConfig(kp=np.array([5.0, 5.0, 2.5])),
            RateControllerConfig(kp=RATE_ACCEL * np.diag(params.inertia)),
        )

    def compute(self, state: np.ndarray, target: np.ndarray, yaw: float, dt: float) -> np.ndarray:
        desired_velocity = self.position.compute(state[:3], target, dt, velocity=state[6:9])
        desired_euler, thrust = self.velocity.compute(state[6:9], yaw, desired_velocity, dt)
        return self.attitude.compute(state[3:6], state[9:12], desired_euler, thrust, dt)


def _fly(preset: VehiclePreset, lane: float, steps: int) -> dict[str, np.ndarray]:
    """Fly one airframe through the box and return its recorded history."""
    aircraft = create_multirotor(preset)
    legs = _waypoints(lane)
    aircraft.reset(position=legs[0].copy())
    aircraft.spin_up_to_hover()

    pilot = _Pilot(aircraft)

    positions = np.zeros((steps, 3))
    eulers = np.zeros((steps, 3))
    targets = np.zeros((steps, 3))
    thrusts = np.zeros((steps, aircraft.n_rotors))
    speeds = np.zeros((steps, aircraft.n_rotors))

    for i in range(steps):
        target, yaw = _command(i * DT, legs)
        wrench = pilot.compute(aircraft.state, target, yaw, DT)
        aircraft.step(wrench, DT)

        positions[i] = aircraft.position
        eulers[i] = aircraft.euler
        targets[i] = target
        thrusts[i] = aircraft.get_rotor_thrusts()
        speeds[i] = aircraft.get_motor_speeds()

    return {
        "aircraft": aircraft,
        "positions": positions,
        "eulers": eulers,
        "targets": targets,
        "thrusts": thrusts,
        "speeds": speeds,
    }


def main() -> None:
    steps = int(LEG_SECONDS * len(_waypoints(0.0)) / DT)
    times = np.arange(steps) * DT

    flights = [
        (preset, colour, label, _fly(preset, lane, steps))
        for preset, lane, colour, label in AIRFRAMES
    ]

    logger = SimLogger("multirotor_mixer", out_dir=Path(__file__).parent, downsample=25)
    logger.log_metadata("algorithm", "Geometry-derived control allocation")
    logger.log_metadata("dt", DT)
    logger.log_metadata("duration", float(times[-1]))
    logger.log_metadata("airframes", [preset.value for preset, _, _, _ in flights])

    for preset, _, _, flight in flights:
        aircraft: Multirotor = flight["aircraft"]
        error = np.linalg.norm(flight["positions"] - flight["targets"], axis=1)
        # Score how well it holds a waypoint, not how fast it slews to the
        # next one: only the last two seconds of each leg count.
        held = np.zeros(steps, dtype=bool)
        leg_steps = int(LEG_SECONDS / DT)
        for end in range(leg_steps, steps + 1, leg_steps):
            held[end - int(2.0 / DT) : end] = True

        logger.log_summary(f"{preset.value}_rotors", aircraft.n_rotors)
        logger.log_summary(f"{preset.value}_allocation_rank", aircraft.mixer.rank)
        logger.log_summary(f"{preset.value}_yaw_authority_m", float(aircraft.mixer.yaw_authority))
        logger.log_summary(f"{preset.value}_held_position_error_m", float(error[held].mean()))
        logger.log_summary(f"{preset.value}_peak_rotor_thrust_N", float(flight["thrusts"].max()))
        logger.log_summary(
            f"{preset.value}_thrust_headroom",
            float(aircraft.params.max_rotor_thrust / flight["thrusts"].max()),
        )
        logger.log_summary(f"{preset.value}_final_yaw_deg", float(np.degrees(flight["eulers"][-1, 2])))

    for i in range(steps):
        logger.log_step(
            t=float(times[i]),
            quad_position=flights[0][3]["positions"][i],
            hex_position=flights[1][3]["positions"][i],
            octo_position=flights[2][3]["positions"][i],
            hex_rotor_thrusts=flights[1][3]["thrusts"][i],
        )
    logger.save()

    # ── render ────────────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="Derived Control Allocation — One Mixer, Four to Eight Rotors",
        world_size=WORLD_SIZE,
        z_max=Z_MAX,
        figsize=(16, 8),
    )
    ax_data = viz.setup_data_axes(
        ylabel="Thrust above the collective [N]",
        title="Hex S550 — what the derived mixer asks of each rotor",
    )
    # Plotted against the instantaneous mean rather than against zero. The
    # collective is a shared offset that swamps the interesting part; what
    # is left after subtracting it *is* the allocation — a left/right split
    # on the roll legs, and an alternating one during the yaw turn.
    hex_thrusts = flights[1][3]["thrusts"]
    hex_split = hex_thrusts - hex_thrusts.mean(axis=1, keepdims=True)
    limit = float(np.abs(hex_split).max()) * 1.15
    ax_data.set_xlim(0, times[-1])
    ax_data.set_ylim(-limit, limit)
    ax_data.axhline(0.0, color="0.6", lw=0.6, ls="--")
    thrust_lines = [ax_data.plot([], [], lw=0.9, label=name)[0] for name in HEX_ROTOR_NAMES]
    ax_data.legend(fontsize=6, loc="upper left", ncol=3)

    for _, colour, label, flight in flights:
        viz.ax_top.plot(
            flight["positions"][:, 0],
            flight["positions"][:, 1],
            color=colour,
            lw=0.7,
            alpha=0.5,
            label=label,
        )
    viz.ax_top.legend(fontsize=6, loc="upper right")

    trails = [viz.create_trail_artists(color=colour) for _, colour, _, _ in flights]

    anim = SimAnimator("multirotor_mixer", out_dir=Path(__file__).parent, dpi=75)
    anim._fig = viz.fig

    skip = max(1, steps // 130)
    frames = list(range(0, steps, skip))
    vehicle_arts: list = []
    title = viz.ax3d.set_title("Derived mixer")

    def update(f: int) -> None:
        k = frames[min(f, len(frames) - 1)]
        clear_vehicle_artists(vehicle_arts)
        clear_vehicle_artists(viz._vehicle_arts_top)

        for (_, colour, _, flight), trail in zip(flights, trails):
            aircraft: Multirotor = flight["aircraft"]
            viz.update_trail(trail, flight["positions"], k)
            R = Multirotor.rotation_matrix(*flight["eulers"][k])
            vehicle_arts.extend(
                draw_multirotor_3d(
                    viz.ax3d,
                    flight["positions"][k],
                    R,
                    aircraft.rotor_positions,
                    aircraft.spin_directions,
                    scale=12.0,
                    arm_color=colour,
                    arm_lw=2.2,
                    motor_size=26.0,
                )
            )
            (dot,) = viz.ax_top.plot(
                flight["positions"][k, 0], flight["positions"][k, 1], "o", color=colour, ms=4
            )
            viz._vehicle_arts_top.append(dot)

        for j, line in enumerate(thrust_lines):
            line.set_data(times[:k], hex_split[:k, j])

        title.set_text(
            f"t={times[k]:5.1f}s   hex collective={hex_thrusts[k].sum():5.1f} N   "
            f"spread={hex_thrusts[k].max() - hex_thrusts[k].min():4.2f} N   "
            f"yaw={np.degrees(flights[1][3]['eulers'][k, 2]):+5.0f}°"
        )

    anim.animate(update, len(frames))
    anim.save()


if __name__ == "__main__":
    main()
