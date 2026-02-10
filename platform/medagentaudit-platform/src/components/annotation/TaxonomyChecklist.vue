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
        <input
          class="mt-1 h-4 w-4 accent-blue-600"
          type="checkbox"
          :checked="selected.includes(item.key)"
          @change="toggle(item.key)"
        />
        <div class="min-w-0">
          <div class="flex items-center gap-2">
            <span class="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">{{ item.key }}</span>
            <span class="text-sm font-medium text-slate-900">{{ item.title }}</span>
          </div>
          <div class="mt-1 text-xs text-slate-600">{{ item.short }}</div>
        </div>
      </label>
    </div>
  </div>
</template>

