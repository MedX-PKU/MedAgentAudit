<script setup lang="ts">
type Variant = 'primary' | 'secondary' | 'ghost'

const props = withDefaults(
  defineProps<{
    variant?: Variant
    title?: string
    disabled?: boolean
    type?: 'button' | 'submit' | 'reset'
  }>(),
  {
    variant: 'ghost',
    title: '',
    disabled: false,
    type: 'button',
  },
)

const base =
  'inline-flex items-center justify-center rounded-lg p-2 text-sm transition focus:outline-none focus:ring-2 focus:ring-blue-500/30 disabled:pointer-events-none disabled:opacity-60'

const variants: Record<Variant, string> = {
  primary: 'bg-blue-600 text-white hover:bg-blue-700',
  secondary: 'bg-white text-slate-900 ring-1 ring-slate-200 hover:bg-slate-50',
  ghost: 'bg-transparent text-slate-700 hover:bg-slate-100 ring-1 ring-transparent',
}
</script>

<template>
  <button :type="props.type" :title="props.title" :disabled="props.disabled" :class="[base, variants[props.variant]]">
    <slot />
  </button>
</template>

