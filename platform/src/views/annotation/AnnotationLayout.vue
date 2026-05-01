<script setup lang="ts">
import { provide, readonly, ref } from 'vue'
import AppShell from '../../components/layout/AppShell.vue'

import { ANNOTATION_DRAWER_KEY, type AnnotationDrawerContext } from '../../components/layout/annotationDrawer'

const isDrawerOpen = ref(false)

const annotationDrawer: AnnotationDrawerContext = {
  isOpen: readonly(isDrawerOpen),
  toggle: () => {
    isDrawerOpen.value = !isDrawerOpen.value
  },
  close: () => {
    isDrawerOpen.value = false
  },
}

provide(ANNOTATION_DRAWER_KEY, annotationDrawer)

const onToggleSidebar = () => {
  annotationDrawer.toggle()
}
</script>

<template>
  <AppShell title="MedAgentAudit / Annotation" content-width="wide" :drawer-open="isDrawerOpen">
    <template #title-left>
      <div data-drawer>
        <button
          type="button"
          class="mr-1 inline-flex h-9 w-9 items-center justify-center rounded-full border text-sm font-semibold transition-colors"
          :class="isDrawerOpen ? 'border-sky-300 bg-sky-50 text-sky-900' : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'"
          :aria-label="isDrawerOpen ? 'Close sidebar' : 'Open sidebar'"
          :title="isDrawerOpen ? 'Close sidebar' : 'Open sidebar'"
          @click="onToggleSidebar"
        >
          <span class="text-lg leading-none">{{ isDrawerOpen ? '‹' : '☰' }}</span>
        </button>
      </div>
    </template>
    <div class="space-y-4 pt-3 sm:pt-4">
      <RouterView />
    </div>
  </AppShell>
</template>
