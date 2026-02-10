<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    leftWidthClass?: string
    rightWidthClass?: string
    stickyTopClass?: string
  }>(),
  {
    leftWidthClass: 'lg:w-[360px]',
    rightWidthClass: 'lg:w-[420px]',
    stickyTopClass: 'top-[76px]',
  },
)
</script>

<template>
  <div class="grid gap-4 lg:grid-cols-[auto_1fr_auto]">
    <aside :class="['h-fit', props.leftWidthClass]">
      <div :class="['sticky', props.stickyTopClass]">
        <slot name="left" />
      </div>
    </aside>

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
