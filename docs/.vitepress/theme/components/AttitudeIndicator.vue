<!-- Erwin Lejeune — 2026-02-23 -->
<script setup lang="ts">
/**
 * Animated attitude indicator — the site's signature motif.
 *
 * Rather than decorating the page with an abstract gradient, this draws
 * the instrument the whole library is ultimately about: an artificial
 * horizon driven by a real (if simplified) phugoid-like oscillation, so
 * the motion reads as an aircraft settling rather than a loop.
 *
 * Everything is inline SVG with CSS-driven values, so it costs no
 * network requests and inherits the page's theme tokens.
 */
import { onMounted, onUnmounted, ref } from 'vue'

const roll = ref(0)
const pitch = ref(0)
const airspeed = ref(35)
const altitude = ref(120)

let frame = 0
let raf: number | undefined
let reduced = false

function animate() {
  frame += 1
  const t = frame / 60

  // Two lightly damped modes at different frequencies: a fast one that
  // looks like the short period, a slow one like the phugoid. Their sum
  // never quite repeats, which keeps the motion from feeling looped.
  const shortPeriod = Math.sin(t * 0.9) * Math.exp(-((t % 24) / 40))
  const phugoid = Math.sin(t * 0.21 + 1.1)

  roll.value = 16 * phugoid + 4 * shortPeriod
  pitch.value = 7 * Math.sin(t * 0.33) + 2 * shortPeriod
  airspeed.value = 35 + 6 * Math.sin(t * 0.21 + 1.1)
  altitude.value = 120 + 14 * Math.sin(t * 0.21 - 0.4)

  raf = requestAnimationFrame(animate)
}

onMounted(() => {
  reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  if (reduced) {
    // Hold a static, slightly banked attitude so the instrument still
    // reads as an aircraft in flight rather than a blank dial.
    roll.value = 12
    pitch.value = 4
    return
  }
  raf = requestAnimationFrame(animate)
})

onUnmounted(() => {
  if (raf !== undefined) cancelAnimationFrame(raf)
})
</script>

<template>
  <div class="ai" role="img" aria-label="Animated artificial horizon showing an aircraft in flight">
    <svg viewBox="0 0 320 320" class="ai__svg">
      <defs>
        <clipPath id="ai-bezel">
          <circle cx="160" cy="160" r="124" />
        </clipPath>
        <linearGradient id="ai-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--uav-sky-500)" />
          <stop offset="100%" stop-color="var(--uav-sky-300)" />
        </linearGradient>
        <linearGradient id="ai-ground" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#334155" />
          <stop offset="100%" stop-color="#0f172a" />
        </linearGradient>
        <linearGradient id="ai-ring" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="var(--uav-hero-1)" />
          <stop offset="55%" stop-color="var(--uav-hero-2)" />
          <stop offset="100%" stop-color="var(--uav-hero-3)" />
        </linearGradient>
        <filter id="ai-glow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="7" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <!-- Horizon ball: rolls with the aircraft, translates with pitch. -->
      <g clip-path="url(#ai-bezel)">
        <g
          :transform="`rotate(${-roll} 160 160) translate(0 ${pitch * 3.4})`"
          class="ai__ball"
        >
          <rect x="-120" y="-180" width="560" height="340" fill="url(#ai-sky)" />
          <rect x="-120" y="160" width="560" height="360" fill="url(#ai-ground)" />
          <line x1="-120" y1="160" x2="440" y2="160" stroke="#e2e8f0" stroke-width="2.5" />

          <!-- Pitch ladder -->
          <g stroke="#e2e8f0" stroke-width="1.6" opacity="0.85">
            <template v-for="deg in [-20, -10, 10, 20]" :key="deg">
              <line
                :x1="deg % 20 === 0 ? 118 : 132"
                :y1="160 - deg * 3.4"
                :x2="deg % 20 === 0 ? 202 : 188"
                :y2="160 - deg * 3.4"
              />
            </template>
          </g>
        </g>
      </g>

      <!-- Bezel -->
      <circle
        cx="160"
        cy="160"
        r="124"
        fill="none"
        stroke="url(#ai-ring)"
        stroke-width="3"
        filter="url(#ai-glow)"
      />
      <circle cx="160" cy="160" r="136" fill="none" stroke="var(--uav-panel-border)" stroke-width="1" />

      <!-- Roll pointer and scale -->
      <g stroke="var(--uav-ink-300)" stroke-width="2" opacity="0.7">
        <template v-for="deg in [-45, -30, -15, 15, 30, 45]" :key="deg">
          <line
            :x1="160 + 116 * Math.sin((deg * Math.PI) / 180)"
            :y1="160 - 116 * Math.cos((deg * Math.PI) / 180)"
            :x2="160 + 124 * Math.sin((deg * Math.PI) / 180)"
            :y2="160 - 124 * Math.cos((deg * Math.PI) / 180)"
          />
        </template>
      </g>
      <polygon
        :transform="`rotate(${-roll} 160 160)`"
        points="160,32 152,46 168,46"
        fill="var(--uav-amber)"
      />

      <!-- Fixed aircraft symbol -->
      <g stroke="var(--uav-amber)" stroke-width="4" stroke-linecap="round" fill="none">
        <line x1="106" y1="160" x2="140" y2="160" />
        <line x1="180" y1="160" x2="214" y2="160" />
        <path d="M148 160 l12 13 l12 -13" />
      </g>
      <circle cx="160" cy="160" r="3.5" fill="var(--uav-amber)" />
    </svg>

    <div class="ai__readout">
      <div class="ai__cell">
        <span class="ai__label">IAS</span>
        <span class="ai__value">{{ airspeed.toFixed(1) }}<em>m/s</em></span>
      </div>
      <div class="ai__cell">
        <span class="ai__label">ALT</span>
        <span class="ai__value">{{ altitude.toFixed(0) }}<em>m</em></span>
      </div>
      <div class="ai__cell">
        <span class="ai__label">ROLL</span>
        <span class="ai__value">{{ roll.toFixed(1) }}<em>deg</em></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ai {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  width: 100%;
  max-width: 360px;
  margin: 0 auto;
}

.ai__svg {
  width: 100%;
  height: auto;
  filter: drop-shadow(0 24px 50px rgb(2 8 23 / 45%));
}

.ai__ball {
  transition: transform 0.08s linear;
}

.ai__readout {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  width: 100%;
  border: 1px solid var(--uav-panel-border);
  border-radius: var(--uav-radius-sm);
  overflow: hidden;
  background: var(--uav-panel-border);
}

.ai__cell {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  padding: 0.55rem 0.7rem;
  background: var(--vp-c-bg);
}

.ai__label {
  font-family: var(--uav-font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.12em;
  color: var(--vp-c-text-2);
}

.ai__value {
  font-family: var(--uav-font-mono);
  font-size: 0.95rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-brand-1);
}

.ai__value em {
  margin-left: 0.22em;
  font-size: 0.62em;
  font-style: normal;
  color: var(--vp-c-text-2);
}
</style>
