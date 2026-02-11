<script setup lang="ts">
const props = defineProps<{
  done: number
  total: number
}>()

const percent = () => {
  if (props.total <= 0) return 0
  return Math.max(0, Math.min(100, Math.round((props.done / props.total) * 100)))
}
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center justify-between text-xs text-slate-600">
      <span>
        Progress: <span class="font-semibold text-slate-900">{{ props.done }}</span> / {{ props.total }}
      </span>
      <span class="font-mono">{{ percent() }}%</span>
    </div>
    <div class="h-2 w-full overflow-hidden rounded-full bg-slate-100 ring-1 ring-slate-200">
      <div class="h-full rounded-full bg-gradient-to-r from-blue-500 to-emerald-400" :style="{ width: `${percent()}%` }" />
    </div>
  </div>
</template>

