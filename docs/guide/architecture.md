# Architecture

```text
uav_sim/
├── vehicles/            Quadrotor (6DOF), Fixed-Wing, VTOL + presets
│   ├── multirotor/      Newton-Euler rigid body, mixer, motor dynamics
│   ├── fixed_wing/      Aerodynamic coefficients, trim solver, presets
│   ├── vtol/            Tilt-rotor, sharing the fixed-wing airframe model
│   ├── components/      Mixer, motor
│   └── footprint.py     Circular / rectangular footprints, swarm envelopes
├── control/             Rate → Attitude → Velocity → Position (cascaded PID)
│   ├── fixed_wing_autopilot.py   Successive loop closure
│   ├── vtol_controller.py        Mode-scheduled transition control
│   └── state_machine.py          ARM → TAKEOFF → HOVER → TRACKING → LAND
├── guidance/            Fixed-wing path following and mission sequencing
│   ├── fixed_wing_paths.py       Straight-line and orbit vector fields
│   └── fixed_wing_mission.py     Waypoints, racetrack, return-to-launch
├── sensors/             GPS, IMU, Lidar 2D/3D, camera, gimbal, rangefinder
├── estimation/          EKF, UKF, complementary filter, particle filter
├── perception/          Occupancy mapping, obstacle detection, visual servoing
├── path_planning/       A*, RRT*, PRM, potential field, coverage
├── path_tracking/       PID, LQR, MPC, pure pursuit, geometric SO(3)
├── trajectory_planning/ Min-snap, polynomial, quintic, Frenet optimal
├── trajectory_tracking/ Feedback linearisation, MPPI, NMPC
├── costmap/             Occupancy grid, inflation, social, footprint layers
├── environment/         World, obstacles, buildings, environment presets
├── swarm/               Reynolds, consensus, virtual structure, leader-follower
├── frames/              ENU/FLU transforms, FLU↔FRD bridge
├── gym/                 Reinforcement-learning environments and trainer
├── visualization/       Three-panel view, data panels, vehicle artists
├── cli/                 The `flybots` command
└── simulations/         40+ runnable demos
```

## The dependency direction

Layers depend downward only. Nothing in `vehicles/` imports from
`control/`, nothing in `control/` imports from `guidance/` or
`simulations/`.

```text
simulations  ──▶  gym  ──┐
     │                   │
     ▼                   ▼
 guidance ──▶ control ──▶ vehicles ──▶ frames
     │                       ▲
     ▼                       │
 estimation ─────────────────┘
 perception
 planning
```

`frames/` sits at the bottom and imports nothing from the package. That is
what lets every model agree on conventions without a circular import.

## Shared abstractions

### `UAVBase`

Owns the state vector, RK4 integration and reset semantics. A subclass
implements `_dynamics(state, control) -> dstate` and declares its
dimensions.

### `AeroCoefficients` and `airframe_wrench`

The aerodynamic model is a **function**, not a class hierarchy:

```python
wrench = airframe_wrench(
    velocity_body_frd=..., rates_body_frd=..., surfaces=...,
    coeffs=..., wing_area=..., wing_span=..., chord=..., rho=...,
)
```

That is why the tilt-rotor can reuse the fixed wing's aerodynamics without
inheriting from it — a VTOL is not a kind of aeroplane, but it does have a
wing, and the wing does not care what is pushing it along.

`aero_wrench` composes `airframe_wrench` with a propeller. The tilt-rotor
calls the former and supplies its own tilting thrust.

### Plugin protocols

`simulations/plugins.py` defines structural protocols — `PathPlannerPlugin`,
`TrackerPlugin`, `EstimatorPlugin`, `PerceptionPlugin` — so algorithms are
swappable inside a simulation without inheritance.

## Simulation layout

Each simulation is a package:

```text
simulations/<category>/<name>/
├── __init__.py
├── __main__.py      calls run.main()
├── run.py           main(): simulate, log, render
├── README.md        problem statement and references
└── <name>.gif       rendered preview (Git LFS)
```

Discovery walks the tree for `run.py`, so adding one requires no registry
edit. `uav_sim/cli/catalogue.py` reads the summary from the module
docstring **as text** rather than by importing, so listing the catalogue
stays fast and cannot be broken by an import error in one simulation.

## Visualization

`ThreePanelViz` gives a 3-D view plus top and side projections.
`SimAnimator` drives the frames and writes the GIF. `vehicle_artists` draws
each airframe; `data_panel` overlays live telemetry.

Every simulation sets the Matplotlib `Agg` backend explicitly, so
everything runs headless.

## Logging

`SimLogger` writes a JSON record next to each simulation: metadata,
per-step state, trace sampling details, and summary metrics. The trace block
records source steps, retained steps, and the downsample factor, so consumers
can interpret a reduced timeseries without knowing how its simulation was run.
That makes runs comparable across commits without re-reading a GIF.

## Testing

Tests assert on **behaviour**, not shape. The most valuable ones state a
physical fact:

```python
def test_trimmed_flight_holds_altitude_open_loop():
    aircraft = create_fixed_wing(preset)
    controls = aircraft.reset_trimmed(altitude=300.0)
    for _ in range(6000):
        aircraft.step(controls, 0.005)
    assert aircraft.state[2] == pytest.approx(300.0, abs=1.0)
```

A shape assertion passes against a model that integrates altitude the wrong
way. This one does not.

## See also

- [Frames and conventions](/guide/conventions)
- [Flight models](/vehicles/)
- [CONTRIBUTING](https://github.com/guilyx/flybots/blob/main/CONTRIBUTING.md)
