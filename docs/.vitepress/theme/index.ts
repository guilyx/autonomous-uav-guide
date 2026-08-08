// Erwin Lejeune — 2026-02-23
import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'

import Layout from './Layout.vue'
import AttitudeIndicator from './components/AttitudeIndicator.vue'
import CardGrid from './components/CardGrid.vue'
import StatBand from './components/StatBand.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    // Registered globally so any markdown page can drop them in.
    app.component('AttitudeIndicator', AttitudeIndicator)
    app.component('CardGrid', CardGrid)
    app.component('StatBand', StatBand)
  },
} satisfies Theme
