# Erwin Lejeune - 2026-02-18
"""Quadrotor reinforcement-learning tasks.

Four tasks of increasing difficulty, all sharing the same 6DOF quadrotor
model and low-level wrench action:

``hover``
    Hold a point in space from a randomised offset. The "hello world" of
    UAV control learning.
``waypoint``
    Fly to a randomly placed goal and stop there.
``trajectory``
    Track a moving reference along a Lissajous figure.
``landing``
    Descend and touch down gently inside a landing pad.

The action is the normalised body wrench ``[thrust, tau_x, tau_y, tau_z]``
in ``[-1, 1]``. Thrust is mapped so that ``0`` is exactly hover thrust,
which means a freshly initialised zero-mean policy starts out roughly
hovering rather than dropping out of the sky — that alone is the difference
between a task that trains in minutes and one that never gets off the
ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flybots.gym.base import EnvConfig, UAVEnv
from flybots.gym.spaces import Box
from flybots.vehicles.multirotor.quadrotor import Quadrotor, QuadrotorParams

__all__ = [
    "QuadrotorEnvConfig",
    "HoverEnv",
    "WaypointEnv",
    "TrajectoryEnv",
    "LandingEnv",
]


@dataclass
class QuadrotorEnvConfig(EnvConfig):
    """Configuration shared by the quadrotor tasks."""

    world_radius: float = 6.0
    """Half-width of the flyable volume [m]. Leaving it ends the episode."""
    max_altitude: float = 12.0
    spawn_position_noise: float = 1.2
    """Std-dev of the initial position offset from the goal [m]."""
    spawn_velocity_noise: float = 0.5
    spawn_attitude_noise: float = 0.15
    """Std-dev of the initial roll/pitch disturbance [rad]."""

    thrust_range: float = 0.6
    """Thrust authority around hover, as a fraction of hover thrust."""
    max_angular_acceleration: float = 12.0
    """Peak commanded angular acceleration [rad/s^2].

    Torque limits are derived from this and the airframe's inertia rather
    than fixed in N.m, so the same number gives comparable handling on a
    27 g Crazyflie and a 3.6 kg Matrice. Setting it too high makes the task
    unlearnable — a random policy flips the aircraft before it can
    discover anything.
    """

    tilt_limit: float = np.radians(75.0)
    """Beyond this the quadrotor is considered to have lost control."""
    success_radius: float = 0.25
    """Distance within which the goal counts as reached [m]."""


class _QuadrotorTask(UAVEnv):
    """Shared plumbing for the quadrotor tasks."""

    def __init__(
        self,
        config: QuadrotorEnvConfig | None = None,
        params: QuadrotorParams | None = None,
    ) -> None:
        self.config: QuadrotorEnvConfig = config or QuadrotorEnvConfig()
        self.vehicle = Quadrotor(params)
        self.goal = np.zeros(3)
        self._hover_thrust = self.vehicle.params.mass * self.vehicle.params.gravity
        super().__init__(self.config)

    # ── spaces ────────────────────────────────────────────────────────────

    @property
    def action_space(self) -> Box:
        return Box(-1.0, 1.0, shape=(4,))

    @property
    def observation_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=(self._observation_size,))

    # ── vehicle plumbing ──────────────────────────────────────────────────

    @property
    def vehicle_position(self) -> NDArray[np.floating]:
        return self.vehicle.state[:3]

    def _spawn(self, position: NDArray[np.floating]) -> None:
        """Reset the quadrotor near ``position`` with a randomised upset."""
        config = self.config
        offset = self.rng.normal(0.0, config.spawn_position_noise, size=3)
        start = position + offset
        start[2] = float(np.clip(start[2], 0.5, config.max_altitude - 1.0))
        euler = np.zeros(3)
        euler[:2] = self.rng.normal(0.0, config.spawn_attitude_noise, size=2)
        euler[2] = self.rng.uniform(-np.pi, np.pi)
        self.vehicle.reset(
            position=start,
            euler=euler,
            velocity=self.rng.normal(0.0, config.spawn_velocity_noise, size=3),
        )

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        wrench = self._action_to_wrench(action)
        for _ in range(self.config.physics_substeps):
            self.vehicle.step(wrench, self.config.physics_dt)

    def _action_to_wrench(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map ``[-1, 1]^4`` to a body wrench, centred on hover thrust.

        Centring on hover is deliberate: it makes the zero action a stable
        one, so an untrained policy explores *around* flight instead of
        having to first discover that thrust exists.
        """
        config = self.config
        thrust = self._hover_thrust * (1.0 + config.thrust_range * float(action[0]))
        torques = self._max_torque * np.asarray(action[1:4], dtype=float)
        return np.concatenate([[max(thrust, 0.0)], torques])

    @property
    def _max_torque(self) -> NDArray[np.floating]:
        """Per-axis torque limit giving the configured angular acceleration."""
        return self.config.max_angular_acceleration * np.diag(self.vehicle.params.inertia)

    # ── observation ───────────────────────────────────────────────────────

    def _observe(self) -> NDArray[np.floating]:
        """Body-frame observation with a full rotation matrix.

        Two choices here matter more than anything else for whether these
        tasks are learnable at all:

        * **Errors are expressed in the body frame.** The action is a body
          wrench, so a body-frame error makes the policy's job a fixed
          mapping. Handing it a world-frame error instead means the network
          has to internally learn the world-to-body rotation before it can
          act on anything, which is most of what makes naive quadrotor RL
          setups fail to leave the ground.
        * **Attitude enters as the rotation matrix**, not Euler angles.
          Euler angles wrap and gimbal-lock; the nine matrix entries are
          smooth and unique everywhere.
        """
        state = self.vehicle.state
        rotation = self.vehicle.rotation_matrix(*state[3:6])
        position_error_body = rotation.T @ (self.goal - state[:3])
        velocity_body = rotation.T @ state[6:9]
        return np.concatenate(
            [
                position_error_body,
                velocity_body,
                rotation.ravel(),
                state[9:12],
            ]
        ).astype(np.float64)

    _observation_size = 18

    # ── termination ───────────────────────────────────────────────────────

    def _out_of_bounds(self) -> bool:
        position = self.vehicle.state[:3]
        config = self.config
        return bool(
            np.abs(position[0]) > config.world_radius
            or np.abs(position[1]) > config.world_radius
            or position[2] > config.max_altitude
        )

    def _lost_control(self) -> bool:
        roll, pitch = self.vehicle.state[3], self.vehicle.state[4]
        return bool(abs(roll) > self.config.tilt_limit or abs(pitch) > self.config.tilt_limit)

    def _terminated(self) -> tuple[bool, str]:
        if not np.all(np.isfinite(self.vehicle.state)):
            return True, "diverged"
        if self.vehicle.state[2] <= 0.05:
            return True, "ground_contact"
        if self._lost_control():
            return True, "tumbled"
        if self._out_of_bounds():
            return True, "out_of_bounds"
        return False, ""

    # ── shared reward pieces ──────────────────────────────────────────────

    @property
    def goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.vehicle.state[:3]))

    def _tracking_reward(self, distance_scale: float = 1.5) -> tuple[float, dict[str, float]]:
        """Dense shaped reward: be near the goal, slow, and level.

        Uses an exponential position term rather than a negative distance.
        A bounded, always-positive term gives a clear gradient near the goal
        and stops the agent from concluding that ending the episode early is
        cheaper than flying.
        """
        state = self.vehicle.state
        distance = self.goal_distance
        position_term = float(np.exp(-((distance / distance_scale) ** 2)))
        speed_penalty = 0.05 * float(np.linalg.norm(state[6:9]))
        tilt_penalty = 0.05 * float(np.linalg.norm(state[3:5]))
        spin_penalty = 0.02 * float(np.linalg.norm(state[9:12]))

        reward = position_term - speed_penalty - tilt_penalty - spin_penalty
        return reward, {
            "position": position_term,
            "speed": -speed_penalty,
            "tilt": -tilt_penalty,
            "spin": -spin_penalty,
            "distance": distance,
        }


class HoverEnv(_QuadrotorTask):
    """Hold a fixed point in space.

    The quadrotor spawns displaced, tilted and moving; the policy has to
    null all three out and stay put.
    """

    def _reset_task(self) -> None:
        self.goal = np.array([0.0, 0.0, 3.0])
        self._spawn(self.goal)

    def _reward(self, action: NDArray[np.floating]) -> tuple[float, dict[str, float]]:
        # A wide Gaussian on purpose: the reward has to still have a usable
        # slope several metres out, or a policy that starts by drifting away
        # sees a flat zero and has nothing to climb.
        reward, breakdown = self._tracking_reward(distance_scale=2.5)
        if self.goal_distance < self.config.success_radius:
            reward += 0.5
            breakdown["precision_bonus"] = 0.5
        return reward, breakdown


class WaypointEnv(_QuadrotorTask):
    """Fly to a randomly placed waypoint and hold it.

    Terminates successfully once the goal is held, so the policy learns to
    arrive *and settle* rather than to fly through at speed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._settled_steps = 0

    def _reset_task(self) -> None:
        radius = self.config.world_radius * 0.6
        self.goal = np.array(
            [
                self.rng.uniform(-radius, radius),
                self.rng.uniform(-radius, radius),
                self.rng.uniform(1.5, self.config.max_altitude * 0.6),
            ]
        )
        self._settled_steps = 0
        start = self.goal + self.rng.normal(0.0, 2.5, size=3)
        start[2] = float(np.clip(start[2], 1.0, self.config.max_altitude - 1.0))
        self.vehicle.reset(
            position=start,
            euler=np.array([0.0, 0.0, self.rng.uniform(-np.pi, np.pi)]),
        )

    def _reward(self, action: NDArray[np.floating]) -> tuple[float, dict[str, float]]:
        reward, breakdown = self._tracking_reward(distance_scale=3.0)
        settled = (
            self.goal_distance < self.config.success_radius
            and float(np.linalg.norm(self.vehicle.state[6:9])) < 0.3
        )
        self._settled_steps = self._settled_steps + 1 if settled else 0
        if settled:
            reward += 1.0
            breakdown["settled_bonus"] = 1.0
        return reward, breakdown

    def _terminated(self) -> tuple[bool, str]:
        # Holding station for a second counts as solving the task.
        if self._settled_steps >= int(1.0 / self.config.dt):
            return True, "goal_reached"
        return super()._terminated()


class TrajectoryEnv(_QuadrotorTask):
    """Track a moving reference along a 3-D Lissajous figure.

    Harder than :class:`WaypointEnv` because the target never stops, so the
    policy has to learn to lead it rather than chase it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._amplitude = np.array([2.5, 2.5, 0.8])
        self._frequency = np.array([0.15, 0.30, 0.20])
        self._phase = np.zeros(3)
        self._centre = np.array([0.0, 0.0, 3.0])

    def _reference(self, t: float) -> NDArray[np.floating]:
        return self._centre + self._amplitude * np.sin(
            2.0 * np.pi * self._frequency * t + self._phase
        )

    def _reset_task(self) -> None:
        self._phase = self.rng.uniform(0.0, 2.0 * np.pi, size=3)
        self.goal = self._reference(0.0)
        self.vehicle.reset(position=self.goal + self.rng.normal(0.0, 0.3, size=3))

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        super()._apply_action(action)
        self.goal = self._reference((self.step_count + 1) * self.config.dt)

    def _observe(self) -> NDArray[np.floating]:
        # Include where the reference is heading, so the policy can lead it.
        t = self.step_count * self.config.dt
        rotation = self.vehicle.rotation_matrix(*self.vehicle.state[3:6])
        lookahead = rotation.T @ (self._reference(t + 0.5) - self.vehicle.state[:3])
        return np.concatenate([super()._observe(), lookahead]).astype(np.float64)

    _observation_size = 21

    def _reward(self, action: NDArray[np.floating]) -> tuple[float, dict[str, float]]:
        return self._tracking_reward(distance_scale=1.0)


class LandingEnv(_QuadrotorTask):
    """Descend and touch down gently on a pad at the origin.

    The reward for touching down scales with how soft and how central the
    landing is, so a controlled descent beats dropping out of the sky even
    though both end the episode.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pad_radius = 0.6
        self.max_touchdown_speed = 1.0

    def _reset_task(self) -> None:
        self.goal = np.zeros(3)
        start = np.array(
            [
                self.rng.uniform(-2.0, 2.0),
                self.rng.uniform(-2.0, 2.0),
                self.rng.uniform(3.0, 6.0),
            ]
        )
        self.vehicle.reset(
            position=start,
            velocity=self.rng.normal(0.0, 0.3, size=3),
        )

    def _reward(self, action: NDArray[np.floating]) -> tuple[float, dict[str, float]]:
        state = self.vehicle.state
        horizontal = float(np.linalg.norm(state[:2]))
        altitude = float(state[2])
        descent = -float(state[8])

        centring = float(np.exp(-((horizontal / 1.0) ** 2)))
        # Reward descending, but only at a sane rate.
        target_descent = float(np.clip(0.4 * altitude, 0.2, 1.5))
        descent_term = float(np.exp(-((descent - target_descent) ** 2)))
        tilt_penalty = 0.1 * float(np.linalg.norm(state[3:5]))

        reward = 0.5 * centring + 0.5 * descent_term - tilt_penalty
        breakdown = {
            "centring": 0.5 * centring,
            "descent": 0.5 * descent_term,
            "tilt": -tilt_penalty,
            "altitude": altitude,
        }

        if altitude <= 0.05:
            speed = float(np.linalg.norm(state[6:9]))
            if horizontal <= self.pad_radius and speed <= self.max_touchdown_speed:
                bonus = 50.0 * (1.0 - speed / self.max_touchdown_speed)
                reward += bonus
                breakdown["touchdown_bonus"] = bonus
            else:
                reward -= 10.0
                breakdown["crash_penalty"] = -10.0
        return reward, breakdown

    def _terminated(self) -> tuple[bool, str]:
        if self.vehicle.state[2] <= 0.05:
            horizontal = float(np.linalg.norm(self.vehicle.state[:2]))
            speed = float(np.linalg.norm(self.vehicle.state[6:9]))
            landed = horizontal <= self.pad_radius and speed <= self.max_touchdown_speed
            return True, "landed" if landed else "crashed"
        if self._lost_control():
            return True, "tumbled"
        if self._out_of_bounds():
            return True, "out_of_bounds"
        return False, ""
