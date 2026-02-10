<script setup lang="ts">
const model = defineModel<string>({ required: true })

const props = withDefaults(
  defineProps<{
    options: { value: string; label: string }[]
    disabled?: boolean
  }>(),
  { disabled: false },
)
</script>

<template>
  <select
    v-model="model"
    :disabled="props.disabled"
    class="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm outline-none ring-blue-500/20 focus:ring-2 disabled:bg-slate-50"
  >
    <option v-for="opt in props.options" :key="opt.value" :value="opt.value">
      {{ opt.label }}
    </option>
  </select>
</template>

