<script setup lang="ts">
import { TAXONOMY, type TaxonomyKey } from '../../domain/taxonomy'

const selected = defineModel<TaxonomyKey[]>({ required: true })

const toggle = (key: TaxonomyKey) => {
  const set = new Set(selected.value)
  if (set.has(key)) set.delete(key)
  else set.add(key)
  selected.value = Array.from(set)
}
</script>

<template>
  <div class="space-y-2">
    <div class="text-sm font-semibold text-slate-900">Taxonomy (multi-select)</div>
    <div class="grid gap-2 md:grid-cols-2">
      <label
        v-for="item in TAXONOMY"
        :key="item.key"
        class="flex cursor-pointer items-start gap-3 rounded-xl border border-slate-200 bg-white p-3 hover:bg-slate-50"
      >
        <div class="min-w-0 flex-1 space-y-1">
          <div class="text-xs font-medium text-slate-500">{{ item.key }}</div>
          <div class="text-sm font-medium text-slate-800">{{ item.title }}</div>
        </div>
        <input
          class="mt-1 h-4 w-4 accent-blue-600"
          type="checkbox"
          :checked="selected.includes(item.key)"
          @change="toggle(item.key)"
        />
      </label>
    </div>
  </div>
</template>
