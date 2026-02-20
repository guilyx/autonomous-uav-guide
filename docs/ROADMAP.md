# Roadmap — Future Evolutions

> Erwin Lejeune — 2026-02-19

This document captures planned improvements, new features, and architectural
evolutions for the Autonomous UAV Guide project. Items are grouped by domain
and roughly prioritised within each section (high → low).

---

## 1. Vehicle Models

### 1.1 Hexacopter / Octocopter Support
- Extend `Quadrotor` to `Multirotor(n_rotors)` with configurable arm
  geometry (X, H, coaxial layouts).
- Derive mixer matrices automatically from rotor positions and spin
  directions instead of the hard-coded 4×4 mixer.
- Add a `VehiclePreset.HEX_S550` and `VehiclePreset.OCTO_X8`.

### 1.2 Improved Aerodynamic Models
- Add blade-element-theory (BET) rotor model for more realistic thrust
  curves at high advance ratios (relevant to VTOL transition).
- Add parasitic + induced drag to the quadrotor fuselage for speed-
  dependent drag losses visible in high-speed trajectory tracking.
- Add ground-effect thrust augmentation when `z < 2 * rotor_radius`.

### 1.3 VTOL Improvements
- Full tilt-wing model (not just tilt-rotor): wing generates lift as
  a function of angle-of-attack and airspeed.
- Transition corridor planner: auto-compute safe tilt schedule given
  wing CL, weight, and desired forward speed.
- Back-transition stall protection.

### 1.4 Fixed-Wing Autonomy
- Full mission waypoint navigation for fixed-wing (orbit, racetrack,
  return-to-launch patterns).
- L1 adaptive guidance law for fixed-wing path following.
- Wind-aware Dubins path planner for fixed-wing.

---

## 2. Control

### 2.1 Adaptive Control
- Model Reference Adaptive Control (MRAC) that learns unmodelled
  dynamics (payload shifts, motor degradation) online.
- L1 adaptive controller for disturbance rejection.

### 2.2 Robust Control
- H-infinity controller synthesis for quadrotor hover.
- Sliding mode controller for aggressive manoeuvres.

### 2.3 Reinforcement Learning
- PPO/SAC agent for aggressive trajectory tracking (replace PID in
  the inner loop during acrobatic manoeuvres).
- Sim-to-real transfer pipeline with domain randomisation on mass,
  inertia, motor time constants, and wind.

### 2.4 Unified Controller Architecture
- Merge `CascadedPIDController` and `FlightController` into a single
  configurable stack to eliminate the current dual-stack confusion.
- Provide a `ControllerFactory` that builds the appropriate stack from
  a YAML config.

---

## 3. State Estimation

### 3.1 Visual-Inertial Odometry (VIO)
- Monocular VIO pipeline: feature tracking → 5-point essential matrix
  → local BA → EKF fusion with IMU.
- Stereo depth estimation for dense mapping.

### 3.2 Multi-Sensor Fusion
- Factor-graph-based estimation (iSAM2-style) replacing the current
  EKF for tighter coupling of GPS, IMU, lidar, and vision.
- Barometer + magnetometer sensor models and fusion.

### 3.3 SLAM Improvements
- Graph-based SLAM (pose-graph optimisation with loop closure).
- Extend EKF-SLAM to handle dynamic landmarks (moving objects).
- 3D occupancy mapping with OctoMap-style octree compression.

---

## 4. Perception

### 4.1 Object Detection & Tracking
- Multi-object tracking (MOT) with Hungarian assignment and Kalman
  prediction for multiple bounding boxes simultaneously.
- Simulated YOLO-style detector with configurable false positive /
  false negative rates and detection noise.

### 4.2 Semantic Mapping
- Label occupancy grid cells with semantic classes (road, building,
  vegetation, water) for mission-aware planning.
- Height-map construction from lidar point clouds.

### 4.3 Depth Estimation
- Simulated depth camera sensor model.
- Point cloud registration (ICP) between successive scans.

### 4.4 Visual Servoing Enhancements
- Image-Based Visual Servoing (IBVS) with proper interaction matrix
  and Jacobian-based velocity mapping (6-DOF).
- Position-Based Visual Servoing (PBVS) using estimated target pose.
- Hybrid VS (switch IBVS↔PBVS based on feature visibility).

---

## 5. Path Planning

### 5.1 Dynamic Re-Planning
- D* Lite for real-time replanning when obstacles are discovered.
- Informed RRT* with path-heuristic focusing.
- Kinodynamic RRT: plan in state space (position + velocity) to
  produce dynamically feasible trajectories directly.

### 5.2 Multi-Agent Planning
- Conflict-Based Search (CBS) for decoupled multi-drone deconfliction.
- Velocity obstacles (VO / ORCA) for reactive collision avoidance in
  dense swarms.

### 5.3 Energy-Aware Planning
- Battery model (voltage sag under load, capacity vs. temperature).
- Energy-optimal path planning that factors in wind, altitude, and
  payload.
- Return-to-home with energy reserve constraint.

---

## 6. Trajectory Planning & Tracking

### 6.1 Time-Optimal Trajectories
- Time-optimal trajectory through waypoints subject to thrust and
  tilt constraints (using convex optimisation or iterative methods).
- CPC (Complementary Progress Constraints) for collision-free
  multi-segment time allocation.

### 6.2 B-Spline Trajectories
- Uniform B-spline trajectory representation (more numerically stable
  than polynomial for long paths).
- Online B-spline deformation for obstacle avoidance (gradient-based
  local replanning à la EGO-Planner).

### 6.3 Corridor-Constrained Planning
- Safe flight corridor generation from 3D occupancy (convex
  decomposition).
- Corridor-constrained minimum-snap optimisation.

### 6.4 MPPI Improvements
- GPU-accelerated rollout sampling (JAX or CUDA kernels) for
  real-time performance at K>1000 samples.
- Colored-noise MPPI for smoother control sequences.
- Covariance adaptation based on cost landscape curvature.

---

## 7. Swarm Algorithms

### 7.1 Task Allocation
- Market-based task allocation (auction algorithm) for heterogeneous
  missions (inspect, deliver, photograph).
- Dynamic task reallocation when an agent fails.

### 7.2 Advanced Formation Control
- Bearing-only formation control (no range sensor needed).
- Formation morphing: smooth transition between different shapes.
- Obstacle-aware formation deformation.

### 7.3 Communication Models
- Realistic communication graph: range-limited, lossy, with latency.
- Consensus under communication delays and packet drops.
- Decentralised SLAM with inter-agent loop closure sharing.

---

## 8. Environment & Simulation

### 8.1 Wind and Weather
- Dryden / Von Kármán turbulence model for stochastic wind gusts.
- Steady-state wind field (configurable direction and speed).
- Rain / fog degradation of sensor performance.

### 8.2 Terrain
- Digital Elevation Model (DEM) terrain for realistic nap-of-the-earth
  flight.
- Terrain-following altitude controller.

### 8.3 Dynamic Obstacles
- Moving obstacles (pedestrians, vehicles) with configurable
  trajectories.
- Pop-up obstacles for reactive avoidance testing.

### 8.4 Multi-Fidelity Simulation
- Real-time 3D visualisation with PyVista or Open3D (replace
  matplotlib for interactive exploration).
- Gazebo / MuJoCo bridge for physics-in-the-loop testing.
- Software-in-the-loop (SITL) with PX4/ArduPilot via MAVLink.

---

## 9. Infrastructure & Tooling

### 9.1 Configuration System
- YAML/TOML-based simulation configs (vehicle params, controller
  gains, world definition, viz settings) instead of hard-coded
  constants in each `run.py`.
- CLI runner: `uav-sim run --config my_scenario.yaml`.

### 9.2 Benchmarking
- Standardised metrics: RMSE, settling time, overshoot, energy
  consumption, completion time.
- Automated benchmark suite that runs all sims and produces a
  comparison table/report.

### 9.3 Testing
- Property-based testing (Hypothesis) for control stability margins
  across parameter ranges.
- Fuzzing of sensor inputs (NaN, extreme values, dropouts).
- Regression tests that compare GIF checksums to detect unexpected
  visual changes.

### 9.4 Documentation
- Sphinx/MkDocs site with API reference auto-generated from
  docstrings.
- Tutorial notebooks (Jupyter) for each domain (control, planning,
  estimation).
- Architecture decision records (ADRs) for major design choices.

### 9.5 CI/CD
- GitHub Actions: run full test suite + linting on every PR.
- Automated GIF regeneration on `main` merges (cached with LFS).
- Release automation with changelog generation (commitizen).

---

## 10. New Simulation Ideas

| Domain | Simulation | Description |
|--------|-----------|-------------|
| Control | `pid_tuning_comparison` | Side-by-side Ziegler-Nichols vs. manual vs. auto-tuned PID |
| Control | `wind_rejection` | Hover in turbulence with/without feedforward compensation |
| Planning | `dynamic_replan` | D* Lite replanning around discovered obstacles |
| Planning | `multi_drone_deconflict` | CBS or ORCA for 4+ drones in shared airspace |
| Trajectory | `bspline_corridor` | B-spline through safe flight corridors |
| Trajectory | `time_optimal` | Fastest path through gates under thrust constraints |
| Estimation | `vio_demo` | Visual-inertial odometry with feature tracks |
| Perception | `multi_target_tracking` | MOT with multiple moving ground targets |
| Perception | `semantic_mapping` | Labelled occupancy grid from simulated camera |
| Swarm | `task_allocation` | Auction-based task assignment for 6+ drones |
| Swarm | `formation_morphing` | Smooth hexagon → line → V transitions |
| Vehicle | `payload_delivery` | Quadrotor picks up, transports, and drops a payload |
| Vehicle | `autorotation` | Emergency landing via autorotation (motor failure) |
| Environment | `urban_delivery` | Full mission in city environment with wind + dynamic obstacles |

---

## 11. Code Quality Debt

- [x] Extract magic numbers (world sizes, cruise altitudes, DT values)
      into shared `simulations/common.py` module (done: figure_8_ref,
      line_to_goal, STANDARD_DURATION, frame_indices, COSTMAP_CMAP)
- [x] Add data panels to all simulations (tracking error, speed, stats)
- [x] Standardize double-integrator dynamics across all swarm simulations
- [x] Remove duplicate waypoint_tracking simulation
- [x] Slow down search algorithm visualization with frame duplication
- [x] Fix LQR/geometric/flight_ops broken controllers
- [ ] Unify `CascadedPIDController` and `FlightController` stacks
- [ ] Remove `ax_side` backward-compatibility shim from `ThreePanelViz`
      and clean all sims that still reference it
- [ ] Replace `fig.add_axes([...])` manual positioning with `gridspec`
      everywhere
- [ ] Add type stubs for `mpl_toolkits.mplot3d` to silence mypy
- [ ] De-duplicate visualization boilerplate across simulation `run.py`
      files (many share identical update-function patterns)
- [ ] Add `__all__` exports to all `__init__.py` files
- [ ] Property-based tests for all controllers (stability across
      parameter ranges)
