<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const props = defineProps<{
  title: string
}>()

const route = useRoute()

const nav = computed(() => [
  { to: '/', label: 'Home' },
  { to: '/annotation/open-coding', label: 'Annotation' },
])

const active = (path: string) => route.path === path || route.path.startsWith(path + '/')
</script>

<template>
  <div class="min-h-screen bg-slate-50 text-slate-900">
    <header class="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
      <div class="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3">
        <div class="font-semibold tracking-tight">
          {{ props.title }}
        </div>
        <div class="flex-1" />
        <nav class="flex items-center gap-1">
          <RouterLink
            v-for="item in nav"
            :key="item.to"
            :to="item.to"
            class="rounded-lg px-3 py-2 text-sm transition hover:bg-slate-100"
            :class="active(item.to) ? 'bg-slate-100 text-slate-900' : 'text-slate-600'"
          >
            {{ item.label }}
          </RouterLink>
        </nav>
      </div>
    </header>

    <main class="mx-auto max-w-6xl px-4 py-6">
      <slot />
    </main>
  </div>
</template>

