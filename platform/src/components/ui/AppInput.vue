<script setup lang="ts">
const model = defineModel<string>({ required: true })

const props = withDefaults(
  defineProps<{
    placeholder?: string
    type?: string
    disabled?: boolean
  }>(),
  {
    placeholder: '',
    type: 'text',
    disabled: false,
  },
)
</script>

<template>
  <input
    v-model="model"
    :type="props.type"
    :placeholder="props.placeholder"
    :disabled="props.disabled"
    class="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm outline-none ring-blue-500/20 placeholder:text-slate-400 focus:ring-2 disabled:bg-slate-50"
  />
</template>

