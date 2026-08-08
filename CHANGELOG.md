# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

## [0.1.0]

Initial release: quadrotor, fixed-wing and VTOL models, planners,
estimators, perception, swarm algorithms and 40 runnable simulations.

[Unreleased]: https://github.com/guilyx/autonomous-uav-guide/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/guilyx/autonomous-uav-guide/releases/tag/v0.1.0
