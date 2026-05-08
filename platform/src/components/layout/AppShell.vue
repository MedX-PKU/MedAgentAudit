<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

import { toBasePath } from '../../lib/assets'

const props = withDefaults(
  defineProps<{
    title?: string
    contentWidth?: 'default' | 'wide'
    /** When true (annotation sidebar open), header and main shift right to make room */
    drawerOpen?: boolean
  }>(),
  {
    title: 'MedAgentAudit',
    contentWidth: 'default',
    drawerOpen: false,
  },
)

const route = useRoute()

const active = (path: string) => route.path === path || route.path.startsWith(path + '/')

const inAnnotation = computed(() => route.path === '/annotation' || route.path.startsWith('/annotation/'))

const contentWidthClass = computed(() => (props.contentWidth === 'wide' ? 'max-w-[1680px]' : 'max-w-7xl'))
const brandIconUrl = toBasePath('branding/medagentaudit-icon.svg')
const pageLabel = computed(() => (props.title === 'MedAgentAudit' ? '' : props.title.replace(/^MedAgentAudit\s*\/\s*/, '')))
const navItemClass = (isActive: boolean) =>
  isActive
    ? 'bg-white text-slate-950 ring-1 ring-slate-200'
    : 'text-slate-500 hover:bg-white hover:text-slate-950'
</script>

<template>
  <div class="min-h-screen bg-[var(--maa-bg)] text-[var(--maa-ink)]">
    <header
      class="sticky top-0 z-50 border-b border-slate-200/80 bg-white/90 px-2.5 transition-[margin] duration-300 ease-out backdrop-blur-xl sm:px-3 lg:px-4"
      :class="props.drawerOpen ? 'lg:ml-[320px]' : 'ml-0'"
    >
      <div
        class="mx-auto flex w-full items-center justify-between gap-4 py-3"
        :class="contentWidthClass"
      >
        <div class="flex min-w-0 items-center gap-3">
          <slot name="title-left" />
          <RouterLink to="/" class="flex min-w-0 items-center gap-3 text-sm font-medium text-slate-900">
            <img :src="brandIconUrl" alt="" class="h-9 w-9 rounded-full bg-slate-950/5 p-1.5" />
            <span class="hidden sm:inline">MedAgentAudit</span>
          </RouterLink>
          <span v-if="pageLabel" class="hidden truncate rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600 md:inline">
            {{ pageLabel }}
          </span>
        </div>

        <nav class="flex shrink-0 items-center gap-1 rounded-full border border-slate-200/80 bg-white/68 p-1">
          <RouterLink
            to="/"
            class="rounded-full px-3 py-2 text-sm font-medium transition sm:px-4"
            :aria-current="active('/') && !inAnnotation ? 'page' : undefined"
            :class="navItemClass(active('/') && !inAnnotation)"
          >
            Home
          </RouterLink>

          <RouterLink
            to="/annotation"
            class="rounded-full px-3 py-2 text-sm font-medium transition sm:px-4"
            :aria-current="inAnnotation ? 'page' : undefined"
            :class="navItemClass(inAnnotation)"
          >
            Annotation
          </RouterLink>
        </nav>
      </div>
    </header>

    <main class="relative transition-[margin] duration-300 ease-out" :class="props.drawerOpen ? 'lg:ml-[320px]' : 'ml-0'">
      <div class="mx-auto w-full px-2.5 pb-14 sm:px-3 sm:pb-20 lg:px-4 xl:px-[1.125rem]" :class="contentWidthClass">
        <slot />
      </div>
    </main>
  </div>
</template>
