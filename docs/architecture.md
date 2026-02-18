# Architecture

## Package Structure

```
src/uav_sim/
├── vehicles/                     # Vehicle dynamics models
│   ├── base.py                   # UAVBase abstract class (RK4 integration)
│   ├── multirotor/
│   │   └── quadrotor.py          # 6DOF quadrotor with motor dynamics
│   ├── fixed_wing/
│   │   └── fixed_wing.py         # Simplified aerodynamic fixed-wing
│   ├── vtol/
│   │   └── tiltrotor.py          # Tilt-rotor with hover/cruise transition
│   ├── components/
│   │   ├── motor.py              # First-order motor model
│   │   └── mixer.py              # Control allocation (wrench → motor RPM)
│   └── payloads/
│       └── gimbal.py             # 2-axis gimbal with rate limiting
│
├── path_tracking/                # Path-following controllers
│   ├── pid_controller.py         # Cascaded PID (position → attitude)
│   ├── lqr_controller.py         # LQR state feedback
│   └── geometric_controller.py   # SE(3) geometric control
│
├── trajectory_tracking/          # Trajectory tracking controllers
│   ├── feedback_linearisation.py # Input-output linearisation
│   └── mppi.py                   # Model Predictive Path Integral
│
├── path_planning/                # Geometric path planning
│   ├── astar_3d.py               # 3D A* on a voxel grid
│   ├── rrt_3d.py                 # RRT and RRT*
│   └── potential_field_3d.py     # Artificial potential field
│
├── trajectory_planning/          # Time-parametrised trajectories
│   ├── min_snap.py               # Minimum-snap (Mellinger & Kumar)
│   └── polynomial_trajectory.py  # Polynomial trajectory generation
│
├── estimation/                   # State estimation
│   ├── ekf.py                    # Extended Kalman Filter
│   ├── ukf.py                    # Unscented Kalman Filter
│   ├── particle_filter.py        # Sequential Monte Carlo
│   └── complementary_filter.py   # Gyro/accel fusion
│
├── environment/                  # Simulation world
│   ├── world.py                  # World container (indoor / outdoor)
│   ├── obstacles.py              # Sphere, Box, Cylinder primitives
│   ├── buildings.py              # Procedural city grid generator
│   └── (dynamic_agents via world.py)
│
├── costmap/                      # Layered costmaps
│   ├── occupancy_grid.py         # 2D/3D binary/probabilistic grid
│   ├── inflation_layer.py        # Distance-based obstacle inflation
│   ├── social_layer.py           # Velocity-dependent dynamic cost
│   └── costmap.py                # Layered costmap compositor
│
├── sensors/                      # Sensor models
│   ├── imu.py                    # 6-axis IMU with bias & noise
│   ├── gps.py                    # GPS with dropout
│   ├── lidar.py                  # 2D lidar with ray-casting
│   ├── camera.py                 # Pinhole camera
│   └── range_finder.py           # Single-beam altimeter
│
├── perception/                   # Perception algorithms
│   ├── occupancy_mapping.py      # Inverse sensor model (log-odds)
│   ├── obstacle_detection.py     # Cluster-based from lidar
│   └── point_cloud.py            # Ranges→cloud, downsample, ground removal
│
├── swarm/                        # Multi-agent algorithms
│   ├── reynolds_flocking.py      # Boids-style separation/alignment/cohesion
│   ├── consensus_formation.py    # Graph-based consensus
│   ├── virtual_structure.py      # Virtual rigid body formation
│   ├── leader_follower.py        # Leader-follower tracking
│   ├── potential_swarm.py        # Potential-field swarm navigation
│   └── coverage.py               # Voronoi-based area coverage
│
└── visualization/                # Plotting and GIF recording
    ├── animator.py               # SimAnimator (headless Agg backend)
    └── plotting.py               # State history plots
```

## Key Design Decisions

### Vehicle Hierarchy

All vehicle types can share the same `UAVBase` interface with `state`, `step()`, `reset()`, and RK4 integration. The existing `Quadrotor` class pre-dates this base (it has its own motor-level dynamics), so it is not forced to inherit `UAVBase` — instead both implement the same duck-typed API. New vehicle types (`FixedWing`, `Tiltrotor`) extend `UAVBase`.

### Path vs Trajectory

- **Path planning** produces geometric waypoints with no time parametrisation (A\*, RRT\*, Potential Field).
- **Trajectory planning** produces time-stamped references with position, velocity, and acceleration (Min-Snap, Polynomial).
- **Path tracking** follows a setpoint or sequence of setpoints (PID, LQR, Geometric).
- **Trajectory tracking** follows a full state reference trajectory (Feedback Linearisation, MPPI).

### Environment & Costmaps

The `World` is the single source of truth for obstacles (static and dynamic). Costmaps are derived views — the `OccupancyGrid` rasterises the world, and additional layers (inflation, social) compose on top.

### Sensors & Perception

Sensors produce noisy measurements from the true state + world. Perception algorithms consume those measurements to build maps or detect obstacles. This clean separation allows swapping sensor models without changing perception code.
