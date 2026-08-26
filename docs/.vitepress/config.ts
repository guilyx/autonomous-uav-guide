// Erwin Lejeune — 2026-02-23
import { defineConfig } from 'vitepress'

const REPO = 'https://github.com/guilyx/flybots'

export default defineConfig({
  title: 'flybots',
  titleTemplate: ':title — flybots',
  description:
    'From-scratch Python implementations of autonomous UAV algorithms. ' +
    'Multirotor, fixed-wing and VTOL flight models, 40+ runnable simulations, ' +
    'and reinforcement-learning environments for teaching a drone to fly.',

  // GitHub Pages serves this as a project site, so assets live under the
  // repository name. Vercel (used for per-PR previews) serves at the domain
  // root instead, and would 404 on every asset with that prefix.
  base: process.env.VERCEL ? '/' : '/flybots/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: [/^https?:\/\/localhost/],

  head: [
    ['meta', { name: 'theme-color', content: '#38bdf8' }],
    ['meta', { name: 'color-scheme', content: 'dark light' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'flybots — autonomous UAV algorithms' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          'Flight models, planners, estimators and RL environments for autonomous UAVs — ' +
          'every algorithm from scratch, with a runnable simulation.',
      },
    ],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    // Inline SVG favicon: an attitude indicator, matching the site's motif.
    [
      'link',
      {
        rel: 'icon',
        type: 'image/svg+xml',
        href:
          "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E" +
          "%3Ccircle cx='16' cy='16' r='15' fill='%230b1220'/%3E" +
          "%3Cpath d='M1 18a15 15 0 0 0 30 0z' fill='%23334155'/%3E" +
          "%3Cpath d='M1 18a15 15 0 0 1 30 0z' fill='%2338bdf8' opacity='.9'/%3E" +
          "%3Cpath d='M6 18h7l3 4 3-4h7' stroke='%23fbbf24' stroke-width='2' fill='none' " +
          "stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E",
      },
    ],
  ],

  markdown: {
    math: true,
    lineNumbers: false,
    theme: { light: 'github-light', dark: 'github-dark-default' },
  },

  sitemap: { hostname: 'https://guilyx.github.io/flybots/' },

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'flybots',

    nav: [
      { text: 'Guide', link: '/guide/getting-started', activeMatch: '/guide/' },
      { text: 'Vehicles', link: '/vehicles/', activeMatch: '/vehicles/' },
      { text: 'Learning', link: '/learning/', activeMatch: '/learning/' },
      { text: 'Atlas', link: '/simulations/', activeMatch: '/simulations/' },
      {
        text: 'More',
        items: [
          { text: 'Conventions', link: '/guide/conventions' },
          { text: 'CLI reference', link: '/guide/cli' },
          { text: 'Roadmap', link: '/ROADMAP' },
          { text: 'Contributing', link: `${REPO}/blob/main/CONTRIBUTING.md` },
          { text: 'Changelog', link: `${REPO}/blob/main/CHANGELOG.md` },
        ],
      },
    ],

    search: {
      provider: 'local',
      options: {
        detailedView: true,
        miniSearch: {
          searchOptions: { fuzzy: 0.2, prefix: true, boost: { title: 4, text: 2 } },
        },
      },
    },

    outline: { level: [2, 3], label: 'On this page' },

    sidebar: {
      '/guide/': [
        {
          text: 'Getting started',
          items: [
            { text: 'Installation', link: '/guide/installation' },
            { text: 'First flight', link: '/guide/getting-started' },
            { text: 'CLI reference', link: '/guide/cli' },
          ],
        },
        {
          text: 'Foundations',
          items: [
            { text: 'Frames and conventions', link: '/guide/conventions' },
            { text: 'Architecture', link: '/guide/architecture' },
          ],
        },
      ],

      '/vehicles/': [
        {
          text: 'Flight models',
          items: [
            { text: 'Overview', link: '/vehicles/' },
            { text: 'Quadrotor', link: '/vehicles/quadrotor' },
            { text: 'Multirotor (N-rotor)', link: '/vehicles/multirotor' },
            { text: 'Fixed-wing', link: '/vehicles/fixed-wing' },
            { text: 'VTOL tilt-rotor', link: '/vehicles/vtol' },
          ],
        },
        {
          text: 'Working with them',
          items: [
            { text: 'Trim and equilibrium', link: '/vehicles/trim' },
            { text: 'Airframe presets', link: '/vehicles/presets' },
            { text: 'Autopilots', link: '/vehicles/autopilots' },
            { text: 'Mission navigation', link: '/vehicles/mission-navigation' },
          ],
        },
      ],

      '/learning/': [
        {
          text: 'Reinforcement learning',
          items: [
            { text: 'Overview', link: '/learning/' },
            { text: 'Environments', link: '/learning/environments' },
            { text: 'Training a policy', link: '/learning/training' },
            { text: 'Designing a task', link: '/learning/design-notes' },
          ],
        },
      ],

      '/simulations/': [
        {
          text: 'Algorithm atlas',
          items: [{ text: 'All simulations', link: '/simulations/' }],
        },
        {
          text: 'Vehicles',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/simulations/vehicles/' },
            { text: 'Quadrotor dynamics', link: '/simulations/vehicles/quadrotor-dynamics' },
            { text: 'Multirotor mixer', link: '/simulations/vehicles/multirotor-mixer' },
            { text: 'Fixed-wing flight', link: '/simulations/vehicles/fixed-wing-flight' },
            { text: 'VTOL transition', link: '/simulations/vehicles/vtol-transition' },
          ],
        },
        {
          text: 'Estimation',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/simulations/estimation/' },
            { text: 'Complementary filter', link: '/simulations/estimation/complementary-filter' },
            { text: 'EKF', link: '/simulations/estimation/ekf' },
            { text: 'UKF', link: '/simulations/estimation/ukf' },
            { text: 'GPS-IMU fusion', link: '/simulations/estimation/gps-imu-fusion' },
            { text: 'Particle filter', link: '/simulations/estimation/particle-filter' },
          ],
        },
        {
          text: 'Control and path tracking',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/simulations/path-tracking/' },
            { text: 'PID hover', link: '/simulations/path-tracking/pid-hover' },
            { text: 'LQR hover', link: '/simulations/path-tracking/lqr-hover' },
            { text: 'LQR path tracking', link: '/simulations/path-tracking/lqr-tracking' },
            { text: 'MPC tracking', link: '/simulations/path-tracking/mpc-tracking' },
            { text: 'Geometric SO(3)', link: '/simulations/path-tracking/geometric-control' },
            { text: 'Pure pursuit 3D', link: '/simulations/path-tracking/pure-pursuit' },
            {
              text: 'Fixed-wing mission',
              link: '/simulations/path-tracking/fixed-wing-mission',
            },
            { text: 'Path smoothing', link: '/simulations/path-tracking/path-smoothing' },
            { text: 'Flight ops demo', link: '/simulations/path-tracking/flight-ops-demo' },
          ],
        },
        {
          text: 'Path planning',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/simulations/path-planning/' },
            { text: 'A* 3D', link: '/simulations/path-planning/astar-3d' },
            { text: 'RRT* 3D', link: '/simulations/path-planning/rrt-star-3d' },
            { text: 'PRM 3D', link: '/simulations/path-planning/prm-3d' },
            { text: 'Potential field 3D', link: '/simulations/path-planning/potential-field-3d' },
            { text: 'Coverage planning', link: '/simulations/path-planning/coverage-planning' },
          ],
        },
        {
          text: 'Trajectory planning',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/simulations/trajectory-planning/' },
            { text: 'Min-snap', link: '/simulations/trajectory-planning/min-snap' },
            { text: 'Polynomial', link: '/simulations/trajectory-planning/polynomial' },
            { text: 'Quintic polynomial', link: '/simulations/trajectory-planning/quintic' },
            { text: 'Frenet optimal', link: '/simulations/trajectory-planning/frenet-optimal' },
          ],
        },
        {
          text: 'Trajectory tracking',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/simulations/trajectory-tracking/' },
            {
              text: 'Feedback linearisation',
              link: '/simulations/trajectory-tracking/feedback-linearisation',
            },
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
            { text: 'Occupancy mapping', link: '/simulations/perception/occupancy-mapping' },
            { text: 'Visual servoing', link: '/simulations/perception/visual-servoing' },
            { text: 'Visual servoing — fixed camera', link: '/simulations/perception/visual-servoing-fixed' },
            { text: 'Visual servoing — gimbal camera', link: '/simulations/perception/visual-servoing-gimbal' },
            { text: 'Sensor suite', link: '/simulations/perception/sensor-suite' },
          ],
        },
        {
          text: 'Sensors',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/simulations/sensors/' },
            { text: 'Gimbal tracking', link: '/simulations/sensors/gimbal-tracking' },
            { text: 'Gimbal bbox tracking', link: '/simulations/sensors/gimbal-bbox-tracking' },
          ],
        },
        {
          text: 'Swarm',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/simulations/swarm/' },
            { text: 'Reynolds flocking', link: '/simulations/swarm/reynolds-flocking' },
            { text: 'Voronoi coverage', link: '/simulations/swarm/voronoi-coverage' },
            { text: 'Leader-follower', link: '/simulations/swarm/leader-follower' },
            { text: 'Consensus formation', link: '/simulations/swarm/consensus-formation' },
            { text: 'Virtual structure', link: '/simulations/swarm/virtual-structure' },
            { text: 'Potential swarm', link: '/simulations/swarm/potential-swarm' },
          ],
        },
        {
          text: 'Environment',
          collapsed: true,
          items: [
            { text: 'Overview', link: '/simulations/environment/' },
            { text: 'Costmap navigation', link: '/simulations/environment/costmap-navigation' },
          ],
        },
      ],
    },

    socialLinks: [{ icon: 'github', link: REPO }],

    editLink: {
      pattern: `${REPO}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub',
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026 Erwin Lejeune',
    },

    docFooter: { prev: 'Previous', next: 'Next' },
  },
})
