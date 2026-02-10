<script setup lang="ts">
const props = defineProps<{
  show: boolean
  message: string
}>()
</script>

<template>
  <div
    v-if="props.show"
    class="fixed bottom-4 right-4 z-50 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 shadow-lg"
  >
    {{ props.message }}
  </div>
</template>

