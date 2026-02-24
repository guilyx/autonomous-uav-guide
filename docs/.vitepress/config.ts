// Erwin Lejeune — 2026-02-23
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Autonomous UAV Guide',
  description: 'From-scratch Python implementations of algorithms for autonomous UAVs',
  base: '/autonomous-quadrotor-guide/',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Architecture', link: '/guide/architecture' },
      { text: 'Simulations', link: '/simulations/' },
    ],
    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Architecture', link: '/guide/architecture' },
        ],
      },
      {
        text: 'Estimation',
        collapsed: false,
        items: [
          { text: 'Complementary Filter', link: '/simulations/estimation/complementary-filter' },
          { text: 'EKF', link: '/simulations/estimation/ekf' },
          { text: 'UKF', link: '/simulations/estimation/ukf' },
          { text: 'GPS-IMU Fusion', link: '/simulations/estimation/gps-imu-fusion' },
          { text: 'Particle Filter', link: '/simulations/estimation/particle-filter' },
        ],
      },
      {
        text: 'Control & Path Tracking',
        collapsed: false,
        items: [
          { text: 'PID Hover', link: '/simulations/path-tracking/pid-hover' },
          { text: 'LQR Hover', link: '/simulations/path-tracking/lqr-hover' },
          { text: 'Flight Ops Demo', link: '/simulations/path-tracking/flight-ops-demo' },
        ],
      },
      {
        text: 'Trajectory Planning',
        collapsed: false,
        items: [
          { text: 'Min-Snap', link: '/simulations/trajectory-planning/min-snap' },
          { text: 'Polynomial', link: '/simulations/trajectory-planning/polynomial' },
          { text: 'Quintic Polynomial', link: '/simulations/trajectory-planning/quintic' },
          { text: 'Frenet Optimal', link: '/simulations/trajectory-planning/frenet-optimal' },
        ],
      },
      {
        text: 'Trajectory Tracking',
        collapsed: false,
        items: [
          { text: 'Feedback Linearisation', link: '/simulations/trajectory-tracking/feedback-linearisation' },
          { text: 'NMPC', link: '/simulations/trajectory-tracking/nmpc' },
          { text: 'MPPI', link: '/simulations/trajectory-tracking/mppi' },
        ],
      },
      {
        text: 'Perception',
        collapsed: false,
        items: [
          { text: 'EKF-SLAM', link: '/simulations/perception/ekf-slam' },
          { text: 'Occupancy Mapping', link: '/simulations/perception/occupancy-mapping' },
          { text: 'Visual Servoing', link: '/simulations/perception/visual-servoing' },
          { text: 'Sensor Suite', link: '/simulations/perception/sensor-suite' },
        ],
      },
      {
        text: 'Sensors',
        collapsed: false,
        items: [
          { text: 'Gimbal Tracking', link: '/simulations/sensors/gimbal-tracking' },
          { text: 'Gimbal BBox Tracking', link: '/simulations/sensors/gimbal-bbox-tracking' },
        ],
      },
      {
        text: 'Swarm',
        collapsed: false,
        items: [
          { text: 'Reynolds Flocking', link: '/simulations/swarm/reynolds-flocking' },
          { text: 'Voronoi Coverage', link: '/simulations/swarm/voronoi-coverage' },
          { text: 'Leader-Follower', link: '/simulations/swarm/leader-follower' },
          { text: 'Consensus Formation', link: '/simulations/swarm/consensus-formation' },
          { text: 'Virtual Structure', link: '/simulations/swarm/virtual-structure' },
          { text: 'Potential Swarm', link: '/simulations/swarm/potential-swarm' },
        ],
      },
      {
        text: 'Environment',
        collapsed: false,
        items: [
          { text: 'Costmap Navigation', link: '/simulations/environment/costmap-navigation' },
        ],
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/guilyx/autonomous-quadrotor-guide' },
    ],
  },
})
