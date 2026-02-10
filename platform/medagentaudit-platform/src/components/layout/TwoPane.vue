<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    rightWidthClass?: string
    stickyTopClass?: string
  }>(),
  {
    rightWidthClass: 'lg:w-[420px]',
    stickyTopClass: 'top-[76px]',
  },
)
</script>

<template>
  <div class="grid gap-4 lg:grid-cols-[1fr_auto]">
    <div class="min-w-0">
      <slot name="main" />
    </div>
    <aside :class="['h-fit', props.rightWidthClass]">
      <div :class="['sticky', props.stickyTopClass]">
        <slot name="right" />
      </div>
    </aside>
  </div>
</template>

