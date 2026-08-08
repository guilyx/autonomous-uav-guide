<!-- Erwin Lejeune — 2026-02-23 -->
<script setup lang="ts">
/**
 * Linked card grid used by the atlas and overview pages.
 *
 * Usage in markdown:
 *   <CardGrid :items="[{ title: 'EKF', meta: 'estimation',
 *                        body: '...', link: '/simulations/estimation/ekf' }]" />
 */
defineProps<{
  items: { title: string; meta?: string; body?: string; link: string; pill?: string }[]
}>()
</script>

<template>
  <div class="uav-grid">
    <a v-for="item in items" :key="item.link" class="uav-card" :href="item.link">
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
