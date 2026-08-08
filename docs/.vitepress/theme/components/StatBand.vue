<!-- Erwin Lejeune — 2026-02-23 -->
<script setup lang="ts">
/**
 * A row of headline numbers, styled like a readout strip.
 *
 * Usage in markdown:
 *   <StatBand :items="[{ value: '40+', label: 'simulations' }]" />
 */
defineProps<{
  items: { value: string; label: string }[]
}>()
</script>

<template>
  <div class="uav-stat-band">
    <div v-for="item in items" :key="item.label" class="uav-stat">
      <div class="uav-stat__value">{{ item.value }}</div>
      <div class="uav-stat__label">{{ item.label }}</div>
    </div>
  </div>
</template>
