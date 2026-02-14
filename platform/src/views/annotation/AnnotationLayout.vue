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
  <AppShell title="MedAgentAudit / Annotation" :drawer-open="isDrawerOpen">
    <template #title-left>
      <div data-drawer>
        <button
          type="button"
          class="mr-2 inline-flex h-9 w-9 items-center justify-center rounded-lg border transition-colors"
        :class="isDrawerOpen ? 'border-slate-300 bg-slate-100 text-slate-800' : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'"
          :aria-label="isDrawerOpen ? 'Close sidebar' : 'Open sidebar'"
          :title="isDrawerOpen ? 'Close sidebar' : 'Open sidebar'"
          @click="onToggleSidebar"
        >
        <span class="text-lg leading-none">{{ isDrawerOpen ? '‹' : '☰' }}</span>
      </button>
      </div>
    </template>
    <div class="space-y-4">
      <RouterView />
    </div>
  </AppShell>
</template>
