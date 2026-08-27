# Erwin Lejeune - 2026-02-18
"""Fixed-wing reinforcement-learning tasks.

Flying a wing is a genuinely different control problem from hovering a
quadrotor: the aircraft cannot stop, it stalls if it slows down, and
altitude, airspeed and heading are all coupled through the same four
surfaces. That makes it a far better test of a learned controller than
hovering, and a much less forgiving one.

``fw-cruise``
    Hold a commanded altitude, airspeed and course from a disturbed start.
``fw-waypoint``
    Fly through a sequence of waypoints without stalling.

Episodes start from a solved trim condition, so the policy learns to
*maintain and steer* flight rather than having to first discover it. The
action is the normalised surface and throttle vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flybots.gym.base import EnvConfig, UAVEnv
from flybots.gym.spaces import Box
from flybots.vehicles.fixed_wing import (
    FixedWingPreset,
    TrimError,
    create_fixed_wing,
)

__all__ = ["FixedWingEnvConfig", "FixedWingCruiseEnv", "FixedWingWaypointEnv"]


@dataclass
class FixedWingEnvConfig(EnvConfig):
    """Configuration for the fixed-wing tasks."""

    dt: float = 0.02
    max_episode_seconds: float = 40.0
    physics_substeps: int = 4

    preset: FixedWingPreset = FixedWingPreset.SKYWALKER_X8
    reference_altitude: float = 120.0
    altitude_band: float = 60.0
    """Altitude command range around the reference [m]."""
    course_band: float = np.radians(90.0)
    """Course command range around the initial heading [rad]."""

    min_altitude: float = 20.0
    """Below this the aircraft has hit the ground."""
    max_altitude: float = 400.0
    stall_grace_seconds: float = 2.0
    """How long the aircraft may stay stalled before the episode ends."""

    altitude_tolerance: float = 10.0
    airspeed_tolerance: float = 2.0
    course_tolerance: float = np.radians(10.0)

    surface_limit: float = np.radians(25.0)


class _FixedWingTask(UAVEnv):
    """Shared plumbing for the fixed-wing tasks."""

    def __init__(self, config: FixedWingEnvConfig | None = None) -> None:
        self.config: FixedWingEnvConfig = config or FixedWingEnvConfig()
        self.aircraft = create_fixed_wing(self.config.preset)
        self.params = self.aircraft.fw_params
        self._trim_controls = np.array([0.0, 0.0, 0.0, 0.5])
        self._stalled_steps = 0
        super().__init__(self.config)

    # ── spaces ────────────────────────────────────────────────────────────

    @property
    def action_space(self) -> Box:
        return Box(-1.0, 1.0, shape=(4,))

    @property
    def observation_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=(self._observation_size,))

    _observation_size = 16

    @property
    def vehicle_position(self) -> NDArray[np.floating]:
        return self.aircraft.state[:3]

    # ── vehicle plumbing ──────────────────────────────────────────────────

    def _spawn_trimmed(self, altitude: float, heading: float) -> None:
        """Start from trim, then upset it so the policy has work to do."""
        try:
            self._trim_controls = self.aircraft.reset_trimmed(altitude=altitude, heading=heading)
        except TrimError:
            state = np.zeros(12)
            state[2] = altitude
            state[5] = heading
            state[6] = self.params.cruise_airspeed
            self.aircraft.reset(state=state)
            self._trim_controls = np.array([0.0, 0.0, 0.0, 0.5])

        state = self.aircraft.state.copy()
        state[3] += self.rng.normal(0.0, np.radians(8.0))
        state[4] += self.rng.normal(0.0, np.radians(4.0))
        state[6] *= 1.0 + self.rng.normal(0.0, 0.06)
        state[7] += self.rng.normal(0.0, 0.5)
        self.aircraft.reset(state=state)
        self._stalled_steps = 0

    def _action_to_controls(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map ``[-1, 1]^4`` to surfaces and throttle, offset from trim.

        Actions are *deltas from the trim solution*, so a zero action flies
        straight and level. The policy only has to learn the correction,
        not rediscover equilibrium flight from scratch.
        """
        limit = self.config.surface_limit
        surfaces = self._trim_controls[:3] + limit * np.asarray(action[:3], dtype=float)
        throttle = self._trim_controls[3] + 0.5 * float(action[3])
        return np.concatenate([surfaces, [throttle]])

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        controls = self._action_to_controls(action)
        for _ in range(self.config.physics_substeps):
            self.aircraft.step(controls, self.config.physics_dt)
        if self.aircraft.is_stalled():
            self._stalled_steps += 1
        else:
            self._stalled_steps = 0

    # ── observation ───────────────────────────────────────────────────────

    @property
    def course(self) -> float:
        """Direction of travel [rad] — not the same as heading in a sideslip."""
        velocity = self.aircraft.velocity
        return float(np.arctan2(velocity[1], velocity[0]))

    def _command_errors(self) -> NDArray[np.floating]:
        """``[altitude error, airspeed error, course error]``, normalised."""
        altitude_error = (self.goal[2] - self.aircraft.state[2]) / self.config.altitude_band
        airspeed_error = (
            self.target_airspeed - self.aircraft.airspeed
        ) / self.params.cruise_airspeed
        raw = self.target_course - self.course
        course_error = float(np.arctan2(np.sin(raw), np.cos(raw))) / np.pi
        return np.array([altitude_error, airspeed_error, course_error])

    def _observe(self) -> NDArray[np.floating]:
        state = self.aircraft.state
        return np.concatenate(
            [
                self._command_errors(),
                self.attitude_features(state[3:6]),
                state[9:12] / 3.0,
                [
                    self.aircraft.alpha,
                    self.aircraft.beta,
                    self.aircraft.airspeed / self.params.cruise_airspeed,
                    float(self.aircraft.is_stalled()),
                ],
            ]
        ).astype(np.float64)

    # ── reward and termination ────────────────────────────────────────────

    def _tracking_reward(self) -> tuple[float, dict[str, float]]:
        config = self.config
        altitude_error = abs(self.goal[2] - self.aircraft.state[2])
        airspeed_error = abs(self.target_airspeed - self.aircraft.airspeed)
        raw = self.target_course - self.course
        course_error = abs(float(np.arctan2(np.sin(raw), np.cos(raw))))

        altitude_term = float(np.exp(-((altitude_error / config.altitude_tolerance) ** 2)))
        airspeed_term = float(np.exp(-((airspeed_error / config.airspeed_tolerance) ** 2)))
        course_term = float(np.exp(-((course_error / config.course_tolerance) ** 2)))
        # Sideslip is uncomfortable, draggy and a sign of uncoordinated flight.
        sideslip_penalty = 0.3 * abs(self.aircraft.beta)
        stall_penalty = 1.0 if self.aircraft.is_stalled() else 0.0

        reward = (
            0.4 * altitude_term
            + 0.3 * airspeed_term
            + 0.3 * course_term
            - sideslip_penalty
            - stall_penalty
        )
        return reward, {
            "altitude": 0.4 * altitude_term,
            "airspeed": 0.3 * airspeed_term,
            "course": 0.3 * course_term,
            "sideslip": -sideslip_penalty,
            "stall": -stall_penalty,
            "altitude_error": altitude_error,
            "course_error_deg": np.degrees(course_error),
        }

    def _reward(self, action: NDArray[np.floating]) -> tuple[float, dict[str, float]]:
        return self._tracking_reward()

    def _terminated(self) -> tuple[bool, str]:
        state = self.aircraft.state
        if not np.all(np.isfinite(state)):
            return True, "diverged"
        if state[2] <= self.config.min_altitude:
            return True, "ground_contact"
        if state[2] >= self.config.max_altitude:
            return True, "out_of_bounds"
        if abs(state[3]) > np.radians(120.0):
            return True, "upset"
        grace_steps = int(self.config.stall_grace_seconds / self.config.dt)
        if self._stalled_steps > grace_steps:
            return True, "stalled"
        return False, ""


class FixedWingCruiseEnv(_FixedWingTask):
    """Hold a commanded altitude, airspeed and course.

    The command is re-randomised part-way through each episode, so the
    policy has to learn to *acquire* a new setpoint, not just to sit on the
    one it started at.
    """

    def _reset_task(self) -> None:
        config = self.config
        heading = self.rng.uniform(-np.pi, np.pi)
        self._spawn_trimmed(config.reference_altitude, heading)
        self._sample_command(heading)
        self._command_switch_step = int(self.config.max_episode_steps * self.rng.uniform(0.4, 0.6))

    def _sample_command(self, around_heading: float) -> None:
        config = self.config
        self.goal = np.array(
            [
                0.0,
                0.0,
                config.reference_altitude
                + self.rng.uniform(-config.altitude_band, config.altitude_band) * 0.5,
            ]
        )
        self.target_airspeed = self.params.cruise_airspeed * self.rng.uniform(0.9, 1.2)
        self.target_course = around_heading + self.rng.uniform(
            -config.course_band, config.course_band
        )

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        super()._apply_action(action)
        if self.step_count + 1 == self._command_switch_step:
            self._sample_command(self.target_course)


class FixedWingWaypointEnv(_FixedWingTask):
    """Fly through a sequence of waypoints without stalling.

    The course command is regenerated every step to point at the active
    waypoint, which turns navigation into the same altitude/airspeed/course
    problem the cruise task poses — but with a moving target.
    """

    def __init__(self, config: FixedWingEnvConfig | None = None) -> None:
        super().__init__(config)
        self.capture_radius = 40.0
        self.waypoints: NDArray[np.floating] = np.zeros((0, 3))
        self.waypoint_index = 0

    def _reset_task(self) -> None:
        config = self.config
        heading = self.rng.uniform(-np.pi, np.pi)
        self._spawn_trimmed(config.reference_altitude, heading)

        # Lay waypoints out ahead of the aircraft, each a moderate turn from
        # the last, so the course is always reachable without a reversal.
        points = []
        position = self.aircraft.state[:3].copy()
        bearing = heading
        for _ in range(4):
            bearing += self.rng.uniform(-np.radians(60.0), np.radians(60.0))
            leg = self.rng.uniform(150.0, 300.0)
            position = position + np.array([leg * np.cos(bearing), leg * np.sin(bearing), 0.0])
            position[2] = config.reference_altitude + self.rng.uniform(-30.0, 30.0)
            points.append(position.copy())

        self.waypoints = np.array(points)
        self.waypoint_index = 0
        self.target_airspeed = self.params.cruise_airspeed
        self._refresh_command()

    def _refresh_command(self) -> None:
        target = self.waypoints[min(self.waypoint_index, len(self.waypoints) - 1)]
        self.goal = target
        delta = target - self.aircraft.state[:3]
        self.target_course = float(np.arctan2(delta[1], delta[0]))

    @property
    def horizontal_distance_to_waypoint(self) -> float:
        target = self.waypoints[min(self.waypoint_index, len(self.waypoints) - 1)]
        return float(np.linalg.norm(target[:2] - self.aircraft.state[:2]))

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        super()._apply_action(action)
        if self.horizontal_distance_to_waypoint < self.capture_radius:
            self.waypoint_index = min(self.waypoint_index + 1, len(self.waypoints))
        if self.waypoint_index < len(self.waypoints):
            self._refresh_command()

    def _reward(self, action: NDArray[np.floating]) -> tuple[float, dict[str, float]]:
        reward, breakdown = self._tracking_reward()
        captured = self.waypoint_index
        breakdown["waypoints_captured"] = float(captured)
        return reward, breakdown

    def _terminated(self) -> tuple[bool, str]:
        if self.waypoint_index >= len(self.waypoints):
            return True, "route_complete"
        return super()._terminated()
