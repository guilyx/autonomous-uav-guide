# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

**Documentation**
- Pages for the five algorithms that had none: geometric SO(3) control,
  LQR path tracking, MPC path tracking, pure pursuit 3D and path
  smoothing. The atlas index previously listed them under "implemented,
  without a dedicated article yet".
- Each new page and every page for an algorithm changed in this release
  carries the *specific* failure mode that was found in it — what the
  symptom looks like, why it happens, and what the corrected numbers are.
  A page that only restates the textbook equation cannot help someone
  debugging a controller that overshoots.

**Promo video**
- Three new scenes covering the families the reel skipped: trajectory
  planning, estimation and perception — nine families in all.
- A closing atlas scene that pages through every simulation in the
  library, six at a time, each tile playing that simulation's own GIF.
  Driven by the catalogue the CLI walks, so it cannot fall behind it.
  The reel is now 78 s.
- The single-plot scenes filled a square region in the middle of a 16:9
  frame; they now fill the frame. The swarm scene follows the flock
  instead of framing a box it crosses in the first two seconds.
- The poster frame is now rendered by `scripts/make_promo.py` alongside
  the video. It used to be a hand-extracted still, which is exactly the
  kind of artefact that keeps showing last year's behaviour after the code
  moves on — as it had.

**Reinforcement learning** (`uav_sim.gym`)
- Six environments following the Gymnasium API without requiring
  Gymnasium: `hover`, `waypoint`, `trajectory`, `landing`, `fw-cruise`
  and `fw-waypoint`.
- Dependency-free trainer with two derivative-free optimisers —
  Augmented Random Search (default) and the Cross-Entropy Method.
- `MLPPolicy` with save/load, linear by default.
- Optional Gymnasium registration as `uav_sim/Hover-v0` and friends, so
  the environments work with any standard RL library.
- Episode and learning-curve rendering.

**Command-line interface**
- `uav-sim` with `list`, `run`, `info`, `envs`, `train`, `play`, `trim`
  and `doctor`.
- `doctor` flies each airframe as a self-check and exits non-zero on
  failure, so it is usable in CI.
- Simulations are discovered by walking the package — adding one requires
  no registry edit.

**Fixed-wing**
- `compute_trim` — numerical trim solver for steady level and climbing
  flight, raising `TrimError` rather than returning a non-equilibrium.
- Four airframe presets: `AEROSONDE`, `SKYWALKER_X8`, `MINI_TRAINER`,
  `CARGO_UAV`. Only the Aerosonde set is published data; the others are
  documented as representative.
- `FixedWingAutopilot` — successive loop closure for altitude, airspeed
  and course, with gains derived from each airframe's control authority.
- Derived envelope quantities: `stall_airspeed`, `aspect_ratio`,
  `wing_loading`, `load_factor`, `flight_path_angle`.

**VTOL**
- `VTOLController` — mode-scheduled hover / transition / cruise /
  back-transition control.
- Rate-limited tilt actuator, `lift_fraction`, `max_torque`.

**Frames**
- `flu_to_frd` / `frd_to_flu` bridge between the library's Forward-Left-Up
  body frame and the Forward-Right-Down frame used by aerodynamics texts.
- `gravity_in_body`, `euler_rates_from_body_rates`.

**Project**
- Documentation site with a unified structure, deployed to GitHub Pages.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR
  templates — the first two were previously empty files.
- `LICENSE` — previously an empty file, despite the README declaring MIT.
- CI matrix over Python 3.12 and 3.13, coverage reporting, a CLI smoke
  test and a docs build check.

### Fixed

**Fixed-wing aerodynamics** — the previous model was not usable for
anything beyond producing a moving picture:

- Integrated `z` **downward** (NED) while every other vehicle in the
  library treats `z` as ENU altitude. Altitude control ran inverted.
- Declared `Cm0`, `Cma` and `e_oswald` but never read them. The aircraft
  had **no static pitch stability** and **no induced drag**.
- No side force, no damping derivatives, no sideslip. The rudder produced
  a yaw moment but no weathercock response, and the short-period mode was
  undamped.
- Thrust was modelled as `throttle * mass * gravity` — no propeller, no
  airspeed dependence, so level flight never settled at a finite speed.
- Control effectiveness used unexplained `0.1` factors.

Replaced with the full Beard & McLain model (Chapter 4, Appendix E):
complete stability and control derivatives, an induced-drag polar,
post-stall flat-plate lift blending, and a momentum-theory propeller.

**VTOL tilt-rotor**

- Angle of attack was computed from the **world-frame** flight path, so
  pitch attitude had no effect on the wing at all.
- Wing lift was gated on **rotor tilt**. Lift comes from airspeed; where
  the rotors point is irrelevant to the wing.
- Lift never rotated with bank, so banking produced no turn and
  coordinated flight was impossible.

**Repository**

- `LICENSE`, `CONTRIBUTING.md` and `.python-version` were empty files.
  `uv sync` failed outright on the empty `.python-version`.
- Removed 17 empty placeholder pages and a parallel README tree that
  shadowed the real documentation.
- Removed the Vercel deployment config in favour of GitHub Pages.

**Trajectory tracking** — every figure-8 tracker was fighting the
reference instead of following it. All six had the reference acceleration
available and none of them used it, so the error was driven by the
reference's own acceleration and each overshot the fast axis of the
figure-8 by up to 18 %.

| tracker | mean error before | after |
|---|---|---|
| Geometric SO(3) | 0.388 m | 0.023 m |
| LQR | 0.158 m | 0.017 m |
| MPC | 0.395 m | 0.032 m |
| Feedback linearisation | 0.234 m | 0.014 m |
| MPPI | 2.190 m | 0.279 m |
| NMPC | 1.033 m | 0.026 m |

- `GeometricController` — attitude gains are now derived from the inertia
  for a stated bandwidth instead of being typed in, so the attitude loop
  is genuinely faster than the position loop; the desired angular
  velocity is recovered by differentiating `R_d` rather than assumed
  zero.
- `LQRController.compute` takes `feedforward_acc`: an LQR is a regulator
  and cannot track an accelerating reference from error alone.
- `MPCController` scores each horizon step against **its own** reference
  and uses the discrete algebraic Riccati solution as terminal cost. It
  previously held one reference point across the horizon (throwing away
  the preview that is the entire reason to run MPC over LQR) and used a
  continuous-time Riccati solution, under-weighting the terminal state by
  a factor of `1/dt`.
- `NMPCTracker` reformulated on thrust and body rates rather than
  torques. Torque moves attitude in milliseconds and position in seconds;
  no single horizon could see both, so the old controller hovered. Also
  gained move blocking, a terminal cost and a horizon-shaped reference —
  and got 3.5× faster.
- `MPPITracker` takes the horizon step in its cost function so it can
  track a moving reference, and exposes `nominal_rollout`. The demo was
  drawing the unweighted mean of the sampled rollouts and calling it the
  optimal trajectory.

**Estimation** — every filter's `Q` was a fixed diagonal, independent of
`dt`. At 200 Hz that claims the velocity random-walks by 0.32 m/s every
5 ms, so each filter discarded its own prediction and echoed the raw
measurement:

- GPS + IMU fusion reported **more** error than IMU-only dead reckoning
  (0.64 m vs 0.52 m). Now 0.42 m against 7.07 m. The IMU also had no
  turn-on bias, which is the term that makes dead reckoning diverge at
  all — white noise alone barely drifts over 30 s.
- The UKF was worse than the EKF on identical inputs (0.82 m vs 0.28 m).
- New `constant_velocity_q` / `constant_acceleration_input_q` build `Q`
  from an acceleration noise density, so the tuning survives a change of
  step size.
- `ParticleFilter` drew from a fresh unseeded generator on every call,
  making runs irreproducible; it now takes a seed, accumulates weights
  multiplicatively and resamples on effective sample size.

**Camera and gimbal**

- `Gimbal.rotation_matrix` returned a matrix with determinant −1 — a
  reflection, not a rotation. Every projected image was mirrored
  horizontally, which silently inverted the sign of every pan-axis
  control law built on it.
- `BBoxTracker` fed normalised device coordinates into a positional
  gimbal command as though they were angles. NDC spans the whole field of
  view over [−1, 1], so the loop's real gain depended on the lens and ran
  far past its stability limit; it is now an angular-rate loop with gains
  in 1/s.
- `VisualServoController` steered *away* from a target off the image
  centre. In gimbal mode the drone had nothing to do at all — the gimbal
  absorbs the image error — so `compute_from_gimbal` closes the position
  loop on the gimbal angles instead.
- `IMU.sense` returned the vehicle's **velocity** labelled as
  acceleration. It now reports specific force in the body frame.

**Swarm**

- Voronoi coverage passed transposed bounds to `CoverageController`,
  producing an empty integration grid, zero force on every agent, and a
  coverage-cost plot that was flat zero for the whole clip. The agents
  never moved. `CoverageController` now rejects the transposed form.
- Reynolds flocking coasted to a standstill (0.19 m/s): all three rules
  vanish once the flock is in formation, so with damping and no
  migratory urge there is nothing left to fly on.
- Virtual structure and leader-follower trailed their formation slots by
  a constant offset — no velocity feed-forward, plus a multiplicative
  velocity decay acting as unmodelled drag on top of the PD's own
  damping. Formation error 2.11 m → 0.16 m and 0.80 m → 0.20 m.
- Leader-follower seeded followers from the global `np.random` state and
  differenced the leader's position against its initial value, producing
  a 250 m/s velocity spike on the first step.

**Vehicles**

- The fixed-wing demo asked a 13.5 kg Aerosonde — which trims near
  35 m/s — to hold 8 m/s inside a 30 m box. It was below stall speed
  from the first frame and hit the ground 2.7 s in. Rebuilt on the trim
  solver and `FixedWingAutopilot` with an airframe and world that match.
- The VTOL demo ramped rotor tilt to 30° and called it a transition. It
  never left rotor-borne flight. Rebuilt on `VTOLController`: full 90°
  tilt, 22 m/s wing-borne cruise carrying 95 % of the weight, and a
  back-transition to hover.

**Simulations**

- `visual_servoing_fixed` and `visual_servoing_gimbal` could not be run
  at all — missing `__init__.py` / `__main__.py`.
- Missions were scored as never having reached their goal: pure pursuit
  stopped tracking at the 1.5 m waypoint threshold while completion was
  checked at 1.0 m, and the loiter was too short to settle the difference.
- Pure pursuit could deadlock on a self-intersecting trajectory, circling
  forever behind its own carrot. The min-snap demo hit its 90 s timeout
  24 m short of the goal; it now finishes in 40 s.
- `OccupancyMapper` integrated every scan as though the vehicle pointed
  east, smearing the map whenever it turned, and painted phantom
  obstacles at the edge of range from noisy no-return beams.
- Blank data panels in the A*, min-snap, Frenet, fixed-wing and VTOL
  demos now plot something.
- Swarm agents rendered as black blobs whatever colour they were given —
  the motor dots were hard-coded black.
- `PotentialField3D` escaped local minima with an unseeded generator.

### Changed

- `FixedWingParams` now composes `AeroCoefficients` and
  `PropulsionParams`. The old flat fields remain as read-only properties,
  so existing code keeps working and the two cannot drift apart.
- `TiltrotorParams` likewise; `CL_alpha` and `CD0` forward to `coeffs`.
- `stall_airspeed` is derived from each model's own lift curve rather
  than being a hand-entered constant.
- Quadrotor: removed a dead duplicate Euler-rate computation.

### Testing

- 76 new tests covering frame conventions, every stability derivative,
  stall behaviour, trim as a genuine equilibrium, autopilot tracking
  across all airframes, and the full VTOL transition sequence.
- Tests assert on flight behaviour rather than array shapes: the headline
  one flies each preset open-loop from trim for thirty seconds and
  requires altitude to hold within a metre.
- 40 more in `test_algorithm_regressions.py`, one per defect found while
  auditing the simulations. Each pins down a specific way an algorithm
  was silently wrong — not wrong enough to crash or fail a shape check,
  but wrong enough that the rendered demo showed the wrong behaviour:
  camera-frame handedness, control-law signs, `dt`-independent process
  noise, missing feed-forward, transposed bounds, RNG determinism.

## [0.1.0]

Initial release: quadrotor, fixed-wing and VTOL models, planners,
estimators, perception, swarm algorithms and 40 runnable simulations.

[Unreleased]: https://github.com/guilyx/autonomous-uav-guide/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/guilyx/autonomous-uav-guide/releases/tag/v0.1.0
