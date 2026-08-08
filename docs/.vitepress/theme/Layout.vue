<!-- Erwin Lejeune — 2026-02-23 -->
<script setup lang="ts">
/**
 * Default layout with the attitude indicator injected into the home hero.
 *
 * VitePress exposes named slots on its layout; using `home-hero-image`
 * keeps the standard hero copy, actions and responsive behaviour while
 * replacing the decorative blob with the site's instrument motif.
 */
import DefaultTheme from 'vitepress/theme'
import AttitudeIndicator from './components/AttitudeIndicator.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #home-hero-image>
      <AttitudeIndicator />
    </template>
  </Layout>
</template>
