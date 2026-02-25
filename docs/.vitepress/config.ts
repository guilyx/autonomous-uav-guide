// Erwin Lejeune — 2026-02-23
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Autonomous UAV Algorithm Handbook',
  description: 'Technical handbook for autonomous UAV algorithms, modeling, and implementation references',
  themeConfig: {
    nav: [],
    outline: false,
    sidebar: [
      {
        text: 'Overview',
        items: [
          { text: 'Handbook Home', link: '/' },
          { text: 'Algorithm Atlas', link: '/simulations/' },
        ],
      },
      {
        text: 'Foundations',
        items: [
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Architecture', link: '/guide/architecture' },
        ],
      },
      {
        text: 'Estimation',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/estimation/' },
          { text: 'Complementary Filter', link: '/simulations/estimation/complementary-filter' },
          { text: 'EKF', link: '/simulations/estimation/ekf' },
          { text: 'UKF', link: '/simulations/estimation/ukf' },
          { text: 'GPS-IMU Fusion', link: '/simulations/estimation/gps-imu-fusion' },
          { text: 'Particle Filter', link: '/simulations/estimation/particle-filter' },
        ],
      },
      {
        text: 'Control and Path Tracking',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/path-tracking/' },
          { text: 'PID Hover', link: '/simulations/path-tracking/pid-hover' },
          { text: 'LQR Hover', link: '/simulations/path-tracking/lqr-hover' },
          { text: 'Flight Ops Demo', link: '/simulations/path-tracking/flight-ops-demo' },
        ],
      },
      {
        text: 'Path Planning',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/path-planning/' },
          { text: 'A* 3D', link: '/simulations/path-planning/astar-3d' },
          { text: 'RRT* 3D', link: '/simulations/path-planning/rrt-star-3d' },
          { text: 'PRM 3D', link: '/simulations/path-planning/prm-3d' },
          { text: 'Potential Field 3D', link: '/simulations/path-planning/potential-field-3d' },
          { text: 'Coverage Planning', link: '/simulations/path-planning/coverage-planning' },
        ],
      },
      {
        text: 'Trajectory Planning',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/trajectory-planning/' },
          { text: 'Min-Snap', link: '/simulations/trajectory-planning/min-snap' },
          { text: 'Polynomial', link: '/simulations/trajectory-planning/polynomial' },
          { text: 'Quintic Polynomial', link: '/simulations/trajectory-planning/quintic' },
          { text: 'Frenet Optimal', link: '/simulations/trajectory-planning/frenet-optimal' },
        ],
      },
      {
        text: 'Trajectory Tracking',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/trajectory-tracking/' },
          { text: 'Feedback Linearisation', link: '/simulations/trajectory-tracking/feedback-linearisation' },
          { text: 'NMPC', link: '/simulations/trajectory-tracking/nmpc' },
          { text: 'MPPI', link: '/simulations/trajectory-tracking/mppi' },
        ],
      },
      {
        text: 'Perception',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/perception/' },
          { text: 'EKF-SLAM', link: '/simulations/perception/ekf-slam' },
          { text: 'Occupancy Mapping', link: '/simulations/perception/occupancy-mapping' },
          { text: 'Visual Servoing', link: '/simulations/perception/visual-servoing' },
          { text: 'Sensor Suite', link: '/simulations/perception/sensor-suite' },
        ],
      },
      {
        text: 'Sensors',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/sensors/' },
          { text: 'Gimbal Tracking', link: '/simulations/sensors/gimbal-tracking' },
          { text: 'Gimbal BBox Tracking', link: '/simulations/sensors/gimbal-bbox-tracking' },
        ],
      },
      {
        text: 'Swarm',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/swarm/' },
          { text: 'Reynolds Flocking', link: '/simulations/swarm/reynolds-flocking' },
          { text: 'Voronoi Coverage', link: '/simulations/swarm/voronoi-coverage' },
          { text: 'Leader-Follower', link: '/simulations/swarm/leader-follower' },
          { text: 'Consensus Formation', link: '/simulations/swarm/consensus-formation' },
          { text: 'Virtual Structure', link: '/simulations/swarm/virtual-structure' },
          { text: 'Potential Swarm', link: '/simulations/swarm/potential-swarm' },
        ],
      },
      {
        text: 'Vehicles',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/vehicles/' },
          { text: 'Quadrotor Dynamics', link: '/simulations/vehicles/quadrotor-dynamics' },
          { text: 'Fixed-Wing Flight', link: '/simulations/vehicles/fixed-wing-flight' },
          { text: 'VTOL Transition', link: '/simulations/vehicles/vtol-transition' },
        ],
      },
      {
        text: 'Environment',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/simulations/environment/' },
          { text: 'Costmap Navigation', link: '/simulations/environment/costmap-navigation' },
        ],
      },
    ],
    socialLinks: [],
  },
})
