<script setup lang="ts">
import type { Ref } from 'vue'
import { computed, onBeforeUnmount, onMounted, provide, readonly, ref } from 'vue'
import { useRoute } from 'vue-router'

import { ANNOTATION_DRAWER_KEY } from './annotationDrawer'
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

const isAnnotationMenuOpen = ref(false)

// Only provide when NOT in annotation routes. Under /annotation/*, AnnotationLayout
// provides the drawer context; we must not override it or views would get the
// wrong one (dropdown menu state instead of sidebar drawer).
const inAnnotationRoute = route.path === '/annotation' || route.path.startsWith('/annotation/')
if (!inAnnotationRoute) {
  provide(ANNOTATION_DRAWER_KEY, {
    isOpen: readonly(isAnnotationMenuOpen) as Ref<boolean>,
    toggle: () => {
      isAnnotationMenuOpen.value = !isAnnotationMenuOpen.value
    },
    close: () => {
      isAnnotationMenuOpen.value = false
    },
  })
}


const active = (path: string) => route.path === path || route.path.startsWith(path + '/')

const inAnnotation = computed(() => route.path === '/annotation' || route.path.startsWith('/annotation/'))

const contentWidthClass = computed(() => (props.contentWidth === 'wide' ? 'max-w-[1680px]' : 'max-w-7xl'))
const brandIconUrl = toBasePath('branding/medagentaudit-icon.svg')
const pageLabel = computed(() => (props.title === 'MedAgentAudit' ? '' : props.title.replace(/^MedAgentAudit\s*\/\s*/, '')))
const navItemClass = (isActive: boolean) =>
  isActive
    ? 'bg-white text-slate-950 ring-1 ring-slate-200'
    : 'text-slate-500 hover:bg-white hover:text-slate-950'

const onDocumentClick = (event: MouseEvent) => {
  if (!isAnnotationMenuOpen.value) return
  const target = event.target as HTMLElement | null
  if (!target) return
  if (target.closest('[data-annotation-menu]')) return
  isAnnotationMenuOpen.value = false
}

onMounted(() => {
  document.addEventListener('click', onDocumentClick)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', onDocumentClick)
})
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

          <div class="relative flex items-center" data-annotation-menu>
            <div class="flex items-center rounded-full transition" :class="navItemClass(inAnnotation)">
              <RouterLink
                to="/annotation"
                class="rounded-l-full px-3 py-2 text-sm font-medium transition sm:px-4"
                :aria-current="route.path === '/annotation' ? 'page' : undefined"
                @click="isAnnotationMenuOpen = false"
              >
                Annotation
              </RouterLink>
              <button
                type="button"
                class="cursor-pointer rounded-r-full border-l border-slate-200/70 px-2 py-2 text-sm font-medium transition hover:bg-slate-50"
                :aria-expanded="isAnnotationMenuOpen"
                aria-haspopup="menu"
                aria-label="Open annotation menu"
                @click="isAnnotationMenuOpen = !isAnnotationMenuOpen"
              >
                <span aria-hidden="true">⌄</span>
              </button>
            </div>
            <div
              v-if="isAnnotationMenuOpen"
              class="absolute right-0 top-full mt-2 w-52 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-[0_20px_50px_rgba(15,23,42,0.14)]"
            >
              <RouterLink
                to="/annotation"
                class="block px-4 py-3 text-sm font-medium hover:bg-slate-50"
                :class="route.path === '/annotation' ? 'bg-sky-50 text-sky-900' : 'text-slate-700'"
                @click="isAnnotationMenuOpen = false"
              >
                Overview
              </RouterLink>
              <RouterLink
                to="/annotation/open-coding"
                class="block border-t border-slate-100 px-4 py-3 text-sm font-medium hover:bg-slate-50"
                :class="active('/annotation/open-coding') ? 'bg-sky-50 text-sky-900' : 'text-slate-700'"
                @click="isAnnotationMenuOpen = false"
              >
                Open-coding
              </RouterLink>
              <RouterLink
                to="/annotation/audit"
                class="block border-t border-slate-100 px-4 py-3 text-sm font-medium hover:bg-slate-50"
                :class="active('/annotation/audit') ? 'bg-sky-50 text-sky-900' : 'text-slate-700'"
                @click="isAnnotationMenuOpen = false"
              >
                Audit
              </RouterLink>
            </div>
          </div>
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
