<script setup lang="ts">
import type { Ref } from 'vue'
import { computed, onBeforeUnmount, onMounted, provide, readonly, ref } from 'vue'
import { useRoute } from 'vue-router'

import { ANNOTATION_DRAWER_KEY } from './annotationDrawer'

const props = defineProps<{
  title: string
  /** When true (annotation sidebar open), header and main shift right to make room */
  drawerOpen?: boolean
}>()

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


const nav = computed(() => [{ to: '/', label: 'Home' }])

const active = (path: string) => route.path === path || route.path.startsWith(path + '/')

const inAnnotation = computed(() => route.path === '/annotation' || route.path.startsWith('/annotation/'))

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
  <div class="min-h-screen bg-slate-50 text-slate-900">
    <header
      class="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur transition-[margin] duration-300 ease-out"
      :class="props.drawerOpen ? 'ml-[320px]' : 'ml-0'"
    >
      <div class="flex w-full items-center gap-4 px-3 py-3 lg:px-4">
        <div class="flex items-center font-semibold tracking-tight">
          <slot name="title-left" />
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

          <div class="relative" data-annotation-menu>
            <button
              type="button"
              class="cursor-pointer rounded-lg px-3 py-2 text-sm transition hover:bg-slate-100"
              :class="inAnnotation ? 'bg-slate-100 text-slate-900' : 'text-slate-600'"
              @click="isAnnotationMenuOpen = !isAnnotationMenuOpen"
            >
              Annotation
            </button>
            <div
              v-if="isAnnotationMenuOpen"
              class="absolute right-0 mt-2 w-44 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg"
            >
              <RouterLink
                to="/annotation/open-coding"
                class="block px-3 py-2 text-sm hover:bg-slate-50 hover:text-slate-900"
                :class="active('/annotation/open-coding') ? 'bg-blue-600 text-white hover:bg-blue-600 hover:text-slate-900' : 'text-slate-700'"
                @click="isAnnotationMenuOpen = false"
              >
                OpenCoding
              </RouterLink>
              <RouterLink
                to="/annotation/audit"
                class="block px-3 py-2 text-sm hover:bg-slate-50 hover:text-slate-900"
                :class="active('/annotation/audit') ? 'bg-blue-600 text-white hover:bg-blue-600 hover:text-slate-900' : 'text-slate-700'"
                @click="isAnnotationMenuOpen = false"
              >
                Audit
              </RouterLink>
            </div>
          </div>
        </nav>
      </div>
    </header>

    <main class="w-full px-3 py-4 lg:px-4 lg:py-5">
      <slot />
    </main>
  </div>
</template>
