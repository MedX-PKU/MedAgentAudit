<script setup lang="ts">
type Variant = 'primary' | 'secondary' | 'ghost'

const props = withDefaults(
  defineProps<{
    type?: 'button' | 'submit' | 'reset'
    variant?: Variant
    disabled?: boolean
  }>(),
  {
    type: 'button',
    variant: 'primary',
    disabled: false,
  },
)

const base =
  'inline-flex items-center justify-center rounded-full px-4 py-2 text-sm font-semibold transition focus:outline-none focus:ring-4 focus:ring-blue-500/20 disabled:pointer-events-none disabled:opacity-60'

const variants: Record<Variant, string> = {
  primary: 'bg-slate-950 text-white shadow-[0_16px_30px_rgba(15,23,42,0.14)] hover:bg-slate-800',
  secondary: 'bg-white text-slate-900 ring-1 ring-slate-200 hover:bg-slate-50',
  ghost: 'bg-transparent text-slate-700 hover:bg-white/70',
}
</script>

<template>
  <button :type="props.type" :disabled="props.disabled" :class="[base, variants[props.variant]]">
    <slot />
  </button>
</template>
