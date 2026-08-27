# Erwin Lejeune - 2026-02-18
"""Tests for the reinforcement-learning environments and trainer."""

import numpy as np
import pytest

from flybots.gym import ENV_SPECS, evaluate, list_envs, make, train
from flybots.gym.base import EnvConfig, UAVEnv
from flybots.gym.optimizers import AugmentedRandomSearch, CrossEntropyMethod, build_optimizer
from flybots.gym.policy import MLPPolicy, RunningNormalizer
from flybots.gym.quadrotor_envs import QuadrotorEnvConfig
from flybots.gym.spaces import Box
from flybots.gym.train import TrainConfig, rollout

ALL_ENV_IDS = sorted(ENV_SPECS)


class TestBox:
    def test_scalar_bounds_broadcast(self):
        space = Box(-1.0, 1.0, shape=(4,))
        assert space.shape == (4,)
        assert space.contains(np.zeros(4))

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="low bound"):
            Box(np.array([1.0]), np.array([-1.0]))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            Box(np.zeros(3), np.ones(4))

    def test_clip_projects_into_the_box(self):
        space = Box(-1.0, 1.0, shape=(3,))
        assert np.allclose(space.clip(np.array([5.0, -5.0, 0.2])), [1.0, -1.0, 0.2])

    def test_sample_is_inside(self):
        space = Box(-2.0, 3.0, shape=(5,))
        rng = np.random.default_rng(0)
        assert space.contains(space.sample(rng))

    def test_infinite_bounds_sample_finitely(self):
        space = Box(-np.inf, np.inf, shape=(4,))
        assert np.all(np.isfinite(space.sample(np.random.default_rng(0))))


class TestRegistry:
    def test_every_spec_builds(self):
        for spec in list_envs():
            env = make(spec.env_id)
            assert isinstance(env, UAVEnv)
            env.close()

    def test_unknown_id_lists_alternatives(self):
        with pytest.raises(KeyError, match="Available"):
            make("does-not-exist")

    def test_seed_reaches_the_env(self):
        env = make("hover", seed=7)
        assert env.config.seed == 7
        env.close()


class TestEnvironmentContract:
    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_reset_returns_matching_observation(self, env_id):
        env = make(env_id, seed=0)
        observation, info = env.reset()
        assert observation.shape == env.observation_space.shape
        assert np.all(np.isfinite(observation))
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_step_returns_the_five_tuple(self, env_id):
        env = make(env_id, seed=0)
        env.reset()
        result = env.step(np.zeros(env.action_space.shape))
        assert len(result) == 5
        observation, reward, terminated, truncated, info = result
        assert observation.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_breakdown" in info
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_episode_terminates_or_truncates(self, env_id):
        env = make(env_id, seed=0)
        env.reset()
        for step in range(env.config.max_episode_steps + 5):
            _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
            if terminated or truncated:
                break
        else:
            pytest.fail("episode never ended")
        assert step < env.config.max_episode_steps + 5
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_terminated_carries_a_reason(self, env_id):
        env = make(env_id, seed=1)
        env.reset()
        rng = np.random.default_rng(1)
        for _ in range(env.config.max_episode_steps):
            action = rng.uniform(-1, 1, size=env.action_space.shape)
            _, _, terminated, truncated, info = env.step(action)
            if terminated:
                assert info["termination_reason"]
                break
            if truncated:
                assert "termination_reason" not in info
                break
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_same_seed_gives_the_same_episode(self, env_id):
        first, second = make(env_id), make(env_id)
        observation_a, _ = first.reset(seed=42)
        observation_b, _ = second.reset(seed=42)
        assert np.allclose(observation_a, observation_b)

        action = np.full(first.action_space.shape, 0.1)
        for _ in range(20):
            step_a = first.step(action)
            step_b = second.step(action)
            assert np.allclose(step_a[0], step_b[0])
            assert step_a[1] == pytest.approx(step_b[1])
        first.close()
        second.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_actions_are_clipped_not_rejected(self, env_id):
        env = make(env_id, seed=0)
        env.reset()
        observation, _, _, _, _ = env.step(np.full(env.action_space.shape, 1e6))
        assert np.all(np.isfinite(observation))
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_state_stays_finite_under_random_actions(self, env_id):
        env = make(env_id, seed=3)
        rng = np.random.default_rng(3)
        for _ in range(3):
            env.reset()
            for _ in range(100):
                action = rng.uniform(-1, 1, size=env.action_space.shape)
                observation, reward, terminated, truncated, _ = env.step(action)
                assert np.all(np.isfinite(observation))
                assert np.isfinite(reward)
                if terminated or truncated:
                    break
        env.close()

    def test_trajectory_tracks_the_flight(self):
        env = make("hover", seed=0)
        env.reset()
        for _ in range(30):
            env.step(np.zeros(4))
        assert env.trajectory.shape == (31, 3)
        env.close()


class TestActionCentring:
    """A zero action must be an equilibrium action."""

    def test_zero_action_is_hover_thrust(self):
        env = make("hover", seed=0)
        wrench = env._action_to_wrench(np.zeros(4))
        weight = env.vehicle.params.mass * env.vehicle.params.gravity
        assert wrench[0] == pytest.approx(weight)
        assert np.allclose(wrench[1:], 0.0)
        env.close()

    def test_zero_action_flies_a_wing_level(self):
        env = make("fw-cruise", seed=0)
        env.reset()
        controls = env._action_to_controls(np.zeros(4))
        assert np.allclose(controls, env._trim_controls)
        env.close()

    def test_torque_limit_scales_with_inertia(self):
        env = make("hover", seed=0)
        expected = env.config.max_angular_acceleration * np.diag(env.vehicle.params.inertia)
        assert np.allclose(env._max_torque, expected)
        env.close()


class TestObservationFrame:
    """Errors must be expressed in the frame the action acts in."""

    def test_position_error_is_body_frame(self):
        env = make("hover", seed=0)
        env.reset()
        # Yaw 90 degrees: a goal due north sits along the body's left axis.
        state = env.vehicle.state.copy()
        state[:3] = [0.0, 0.0, 3.0]
        state[3:6] = [0.0, 0.0, np.pi / 2]
        state[6:] = 0.0
        env.vehicle.state = state
        env.goal = np.array([0.0, 2.0, 3.0])

        observation = env._observe()
        assert observation[0] == pytest.approx(2.0, abs=1e-6)  # forward
        assert observation[1] == pytest.approx(0.0, abs=1e-6)
        env.close()

    def test_attitude_enters_as_a_rotation_matrix(self):
        env = make("hover", seed=0)
        env.reset()
        rotation = env._observe()[6:15].reshape(3, 3)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        env.close()


class TestRewards:
    def test_hover_rewards_being_on_the_goal(self):
        env = make("hover", seed=0)
        env.reset()
        env.vehicle.reset(position=env.goal.copy())
        on_goal, _ = env._reward(np.zeros(4))
        env.vehicle.reset(position=env.goal + np.array([4.0, 0.0, 0.0]))
        far_away, _ = env._reward(np.zeros(4))
        assert on_goal > far_away
        env.close()

    def test_position_reward_is_bounded_and_positive(self):
        """An always-positive term stops crashing early being the cheap option."""
        env = make("hover", seed=0)
        env.reset()
        for offset in (0.0, 2.0, 20.0):
            env.vehicle.reset(position=env.goal + np.array([offset, 0.0, 0.0]))
            _, breakdown = env._reward(np.zeros(4))
            assert 0.0 <= breakdown["position"] <= 1.0
        env.close()

    def test_action_cost_penalises_effort_and_chatter(self):
        env = make("hover", seed=0)
        env.reset()
        assert env._action_cost(np.zeros(4)) == pytest.approx(0.0)
        assert env._action_cost(np.ones(4)) < 0.0
        env.close()

    def test_landing_pays_more_for_a_soft_touchdown(self):
        env = make("landing", seed=0)
        env.reset()
        env.vehicle.reset(position=np.array([0.0, 0.0, 0.04]), velocity=np.zeros(3))
        gentle, _ = env._reward(np.zeros(4))
        env.vehicle.reset(position=np.array([0.0, 0.0, 0.04]), velocity=np.array([0.0, 0.0, -0.9]))
        firm, _ = env._reward(np.zeros(4))
        assert gentle > firm
        env.close()

    def test_landing_outside_the_pad_is_a_crash(self):
        env = make("landing", seed=0)
        env.reset()
        env.vehicle.reset(position=np.array([5.0, 0.0, 0.04]), velocity=np.zeros(3))
        done, reason = env._terminated()
        assert done and reason == "crashed"
        env.close()


class TestFixedWingTasks:
    def test_starts_from_trim(self):
        env = make("fw-cruise", seed=0)
        env.reset()
        assert env.aircraft.airspeed > env.params.stall_airspeed
        env.close()

    def test_course_is_the_velocity_direction(self):
        env = make("fw-cruise", seed=0)
        env.reset()
        velocity = env.aircraft.velocity
        assert env.course == pytest.approx(np.arctan2(velocity[1], velocity[0]))
        env.close()

    def test_persistent_stall_ends_the_episode(self):
        env = make("fw-cruise", seed=0)
        env.reset()
        env._stalled_steps = int(env.config.stall_grace_seconds / env.config.dt) + 1
        done, reason = env._terminated()
        assert done and reason == "stalled"
        env.close()

    def test_waypoint_route_advances(self):
        env = make("fw-waypoint", seed=0)
        env.reset()
        assert len(env.waypoints) == 4
        assert env.waypoint_index == 0
        env.close()


class TestPolicy:
    def test_output_layer_starts_at_zero(self):
        """A fresh policy must emit the equilibrium action."""
        for hidden in [(), (16,), (32, 32)]:
            policy = MLPPolicy(18, 4, hidden_sizes=hidden, seed=0)
            assert np.allclose(policy.act(np.random.default_rng(0).normal(size=18)), 0.0)

    def test_actions_stay_in_range(self):
        policy = MLPPolicy(18, 4, hidden_sizes=(32,), seed=0)
        policy.parameters = np.random.default_rng(0).normal(0, 5, policy.parameter_count)
        for _ in range(20):
            action = policy.act(np.random.default_rng(1).normal(size=18) * 10)
            assert np.all(np.abs(action) <= 1.0)

    def test_linear_policy_is_small(self):
        assert MLPPolicy(18, 4, hidden_sizes=()).parameter_count == 18 * 4 + 4

    def test_roundtrip_through_disk(self, tmp_path):
        policy = MLPPolicy(18, 4, hidden_sizes=(16,), seed=0)
        policy.parameters = np.random.default_rng(0).normal(size=policy.parameter_count)
        path = policy.save(tmp_path / "policy.npz")

        loaded = MLPPolicy.load(path)
        assert loaded.hidden_sizes == policy.hidden_sizes
        observation = np.random.default_rng(2).normal(size=18)
        assert np.allclose(loaded.act(observation), policy.act(observation))

    def test_explicit_parameters_override_stored_ones(self):
        policy = MLPPolicy(18, 4, hidden_sizes=(), seed=0)
        other = np.random.default_rng(0).normal(size=policy.parameter_count)
        observation = np.ones(18)
        assert not np.allclose(policy.act(observation, other), policy.act(observation))


class TestRunningNormalizer:
    def test_tracks_mean_and_variance(self):
        rng = np.random.default_rng(0)
        data = rng.normal(3.0, 2.0, size=(4000, 5))
        normalizer = RunningNormalizer(5)
        for chunk in np.array_split(data, 20):
            normalizer.update(chunk)
        assert np.allclose(normalizer.mean, data.mean(axis=0), atol=0.05)
        assert np.allclose(np.sqrt(normalizer.var), data.std(axis=0), atol=0.1)

    def test_empty_batch_is_a_noop(self):
        normalizer = RunningNormalizer(3)
        before = normalizer.mean.copy()
        normalizer.update(np.zeros((0, 3)))
        assert np.allclose(normalizer.mean, before)


class TestOptimizers:
    @staticmethod
    def _quadratic(target):
        def score(parameters):
            return -float(np.sum((parameters - target) ** 2))

        return score

    def test_ars_descends_a_quadratic(self):
        target = np.array([1.0, -2.0, 0.5])
        optimizer = AugmentedRandomSearch(
            np.zeros(3), step_size=0.15, noise=0.2, directions=12, seed=0
        )
        score = self._quadratic(target)
        start = score(optimizer.parameters)
        for _ in range(120):
            optimizer.step(score)
        assert score(optimizer.parameters) > start

    def test_cem_descends_a_quadratic(self):
        target = np.array([1.0, -2.0, 0.5])
        optimizer = CrossEntropyMethod(np.zeros(3), population=40, initial_std=1.0, seed=0)
        score = self._quadratic(target)
        start = score(optimizer.parameters)
        for _ in range(40):
            optimizer.step(score)
        assert score(optimizer.parameters) > start

    def test_ars_survives_a_flat_landscape(self):
        """No gradient information must widen the search, not divide by zero."""
        optimizer = AugmentedRandomSearch(np.zeros(3), directions=6, seed=0)
        noise_before = optimizer.noise
        stats = optimizer.step(lambda _: 1.0)
        assert optimizer.noise > noise_before
        assert np.all(np.isfinite(optimizer.parameters))
        assert stats.evaluations == 12

    def test_build_optimizer_by_name(self):
        assert isinstance(build_optimizer("ars", np.zeros(3)), AugmentedRandomSearch)
        assert isinstance(build_optimizer("CEM", np.zeros(3)), CrossEntropyMethod)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer("adam", np.zeros(3))


class TestTraining:
    def test_short_run_completes(self):
        result = train("hover", iterations=3, directions=4, seed=0)
        assert result.env_id == "hover"
        assert len(result.history) == 3
        assert np.isfinite(result.best_return)
        assert result.total_episodes > 0

    def test_saves_a_loadable_policy(self, tmp_path):
        path = tmp_path / "policy.npz"
        train("hover", iterations=2, directions=2, seed=0, save_path=path)
        assert MLPPolicy.load(path).parameter_count > 0

    def test_progress_callback_fires_each_iteration(self):
        rows = []
        train("hover", iterations=3, directions=2, seed=0, progress=rows.append)
        assert len(rows) == 3
        assert {"iteration", "mean_return", "holdout_return"} <= set(rows[0])

    def test_cem_backend_runs(self):
        result = train(
            "hover",
            config=TrainConfig(optimizer="cem", iterations=2, population=6, seed=0),
        )
        assert np.isfinite(result.best_return)

    def test_training_is_reproducible(self):
        first = train("hover", iterations=3, directions=4, seed=11)
        second = train("hover", iterations=3, directions=4, seed=11)
        assert np.allclose(first.policy.parameters, second.policy.parameters)

    def test_fixed_task_seeds_make_scores_repeatable(self):
        """The objective must be deterministic in the parameters."""
        env = make("hover", seed=0)
        policy = MLPPolicy(18, 4, hidden_sizes=(), seed=0)
        first, _, _ = rollout(env, policy, seed=5)
        second, _, _ = rollout(env, policy, seed=5)
        assert first == pytest.approx(second)
        env.close()

    def test_evaluate_reports_termination_mix(self):
        policy = MLPPolicy(18, 4, hidden_sizes=(), seed=0)
        summary = evaluate("hover", policy, episodes=4, seed=0)
        assert summary["episodes"] == 4
        assert np.isfinite(summary["mean_return"])
        percentages = [v for k, v in summary.items() if k.startswith("pct_")]
        assert percentages and sum(percentages) == pytest.approx(100.0)


class TestEnvConfig:
    def test_step_count_follows_from_dt(self):
        config = EnvConfig(dt=0.02, max_episode_seconds=10.0)
        assert config.max_episode_steps == 500

    def test_physics_substeps_divide_the_control_period(self):
        config = EnvConfig(dt=0.02, physics_substeps=4)
        assert config.physics_dt == pytest.approx(0.005)

    def test_custom_config_reaches_the_env(self):
        env = make("hover", config=QuadrotorEnvConfig(max_episode_seconds=2.0))
        assert env.config.max_episode_steps == 100
        env.close()


class TestGymnasiumAdapter:
    """`gym.make` must yield a conforming environment, not merely register."""

    def test_make_returns_a_working_env(self):
        gymnasium = pytest.importorskip("gymnasium")
        import flybots.gym  # noqa: F401  — registers on import

        env = gymnasium.make("flybots/Hover-v0")
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_adapter_subclasses_gymnasium_env(self):
        """Gymnasium type-checks the entry point and refuses anything else."""
        gymnasium = pytest.importorskip("gymnasium")
        from flybots.gym.registry import _adapter_class

        assert issubclass(_adapter_class(), gymnasium.Env)

    def test_observations_match_the_declared_dtype(self):
        """A widened dtype silently corrupts anything trusting the space."""
        gymnasium = pytest.importorskip("gymnasium")
        import flybots.gym  # noqa: F401

        env = gymnasium.make("flybots/Hover-v0")
        obs, _ = env.reset(seed=0)
        assert obs.dtype == env.observation_space.dtype
        env.close()

    def test_ids_use_the_flybots_namespace(self):
        """Renamed in 2.0.0; the old namespace must be gone entirely."""
        gymnasium = pytest.importorskip("gymnasium")
        import flybots.gym  # noqa: F401

        assert any(k.startswith("flybots/") for k in gymnasium.registry)
        assert not any(k.startswith("uav_sim/") for k in gymnasium.registry)
