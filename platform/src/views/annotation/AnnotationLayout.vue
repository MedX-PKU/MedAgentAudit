<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

import AppShell from '../../components/layout/AppShell.vue'

const route = useRoute()

const tabs = computed(() => [
  { to: '/annotation/open-coding', label: 'Open-coding' },
  { to: '/annotation/audit', label: 'Audit' },
])

const active = (to: string) => route.path === to
</script>

<template>
  <AppShell title="MedAgentAudit / Annotation">
    <div class="space-y-4">
      <div class="flex flex-wrap items-center gap-2">
        <RouterLink
          v-for="tab in tabs"
          :key="tab.to"
          :to="tab.to"
          class="rounded-lg px-3 py-2 text-sm font-medium transition"
          :class="active(tab.to) ? 'bg-blue-600 text-white' : 'bg-white text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50'"
        >
          {{ tab.label }}
        </RouterLink>
      </div>

      <RouterView />
    </div>
  </AppShell>
</template>
