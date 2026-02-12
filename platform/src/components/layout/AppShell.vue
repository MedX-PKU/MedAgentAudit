<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'

const props = defineProps<{
  title: string
}>()

const route = useRoute()

const isAnnotationMenuOpen = ref(false)

const toggleAnnotationMenu = () => {
  isAnnotationMenuOpen.value = !isAnnotationMenuOpen.value
}

const closeAnnotationMenu = () => {
  isAnnotationMenuOpen.value = false
}

const nav = computed(() => [{ to: '/', label: 'Home' }])

const active = (path: string) => route.path === path || route.path.startsWith(path + '/')

const inAnnotation = computed(() => route.path === '/annotation' || route.path.startsWith('/annotation/'))

const onDocumentClick = (event: MouseEvent) => {
  if (!isAnnotationMenuOpen.value) return
  const target = event.target as HTMLElement | null
  if (!target) return
  if (target.closest('[data-annotation-menu]')) return
  closeAnnotationMenu()
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
    <header class="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
      <div class="flex w-full items-center gap-4 px-3 py-3 lg:px-4">
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

          <div class="relative" data-annotation-menu>
            <button
              type="button"
              class="cursor-pointer rounded-lg px-3 py-2 text-sm transition hover:bg-slate-100"
              :class="inAnnotation ? 'bg-slate-100 text-slate-900' : 'text-slate-600'"
              @click="toggleAnnotationMenu"
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
                @click="closeAnnotationMenu"
              >
                OpenCoding
              </RouterLink>
              <RouterLink
                to="/annotation/audit"
                class="block px-3 py-2 text-sm hover:bg-slate-50 hover:text-slate-900"
                :class="active('/annotation/audit') ? 'bg-blue-600 text-white hover:bg-blue-600 hover:text-slate-900' : 'text-slate-700'"
                @click="closeAnnotationMenu"
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
