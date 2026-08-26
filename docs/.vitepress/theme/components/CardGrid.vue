<!-- Erwin Lejeune — 2026-02-23 -->
<script setup lang="ts">
import { withBase } from 'vitepress'

/**
 * Linked card grid used by the atlas and overview pages.
 *
 * Usage in markdown:
 *   <CardGrid :items="[{ title: 'EKF', meta: 'estimation',
 *                        body: '...', link: '/simulations/estimation/ekf' }]" />
 *
 * `link` is written site-absolute and passed through `withBase`. VitePress
 * rewrites links written in markdown to include `base`, but not an href bound
 * in a component, so a raw `:href` sends every card to the domain root. That
 * is invisible on Vercel, where `base` is '/', and 404s on GitHub Pages, where
 * it is '/flybots/'.
 */
defineProps<{
  items: { title: string; meta?: string; body?: string; link: string; pill?: string }[]
}>()
</script>

<template>
  <div class="uav-grid">
    <a v-for="item in items" :key="item.link" class="uav-card" :href="withBase(item.link)">
      <div v-if="item.meta" class="uav-card__meta">{{ item.meta }}</div>
      <div class="uav-card__title">
        {{ item.title }}
        <span v-if="item.pill" :class="['uav-pill', `uav-pill--${item.pill}`]">
          {{ item.pill }}
        </span>
      </div>
      <div v-if="item.body" class="uav-card__body">{{ item.body }}</div>
    </a>
  </div>
</template>
