<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { TAXONOMY, type TaxonomyKey } from '../../../domain/taxonomy'
import AppButton from '../../../components/ui/AppButton.vue'
import AppCard from '../../../components/ui/AppCard.vue'
import AppSelect from '../../../components/ui/AppSelect.vue'
import type { AuditAnnotation, AuditCase, AuditItem } from '../../../domain/types'
import { downloadJson } from '../../../lib/download'
import { AUDIT_CASES } from '../../../data/audit/cases'
import { assignedAuditItems, type AuditorId } from './auditAssignment'
import { loadAuditMap, saveAudit } from './auditStorage'

type Verdict = 'yes' | 'no'

const auditorIdText = ref('1')
const auditorId = computed<AuditorId | null>(() => {
  const n = Number(auditorIdText.value)
  if (![1, 2, 3, 4, 5, 6].includes(n)) return null
  return n as AuditorId
})

const search = ref('')
const activeAuditId = ref<string | null>(null)
const verdict = ref<Verdict>('no')

const annotations = ref<Record<string, AuditAnnotation>>({})

watch(
  auditorId,
  (id) => {
    if (!id) {
      annotations.value = {}
      activeAuditId.value = null
      return
    }
    annotations.value = loadAuditMap(id)
    const items = assignedAuditItems(id, AUDIT_CASES)
    if (!activeAuditId.value && items.length > 0) activeAuditId.value = items[0]!.auditId
  },
  { immediate: true },
)

const assignedItems = computed<AuditItem[]>(() => {
  if (!auditorId.value) return []
  return assignedAuditItems(auditorId.value, AUDIT_CASES)
})

const filteredItems = computed(() => {
  const q = search.value.trim().toLowerCase()
  if (!q) return assignedItems.value
  return assignedItems.value.filter((it) => {
    const hay = [it.auditId, it.caseId, it.taxonomyKey].join(' ').toLowerCase()
    return hay.includes(q)
  })
})

const activeItem = computed<AuditItem | null>(() => {
  if (!activeAuditId.value) return null
  return assignedItems.value.find((i) => i.auditId === activeAuditId.value) ?? null
})

const activeCase = computed<AuditCase | null>(() => {
  if (!activeItem.value) return null
  return AUDIT_CASES.find((c) => c.caseId === activeItem.value!.caseId) ?? null
})

const activeTaxonomy = computed(() => {
  const key = activeItem.value?.taxonomyKey
  if (!key) return null
  return TAXONOMY.find((t) => t.key === key) ?? null
})

watch(
  activeItem,
  (it) => {
    if (!it || !auditorId.value) return
    const existing = annotations.value[it.auditId]
    verdict.value = existing?.verdict ?? 'no'
  },
  { immediate: true },
)

const doneCount = computed(() => Object.keys(annotations.value).length)

const persistActive = () => {
  if (!auditorId.value || !activeItem.value) return
  const annotation: AuditAnnotation = {
    auditId: activeItem.value.auditId,
    caseId: activeItem.value.caseId,
    taxonomyKey: activeItem.value.taxonomyKey as TaxonomyKey,
    verdict: verdict.value,
    updatedAt: new Date().toISOString(),
  }
  saveAudit(auditorId.value, annotation)
  annotations.value = loadAuditMap(auditorId.value)
}

watch(verdict, persistActive)

const exportJson = () => {
  if (!auditorId.value) return
  const name = `Auditor_${auditorId.value}`
  const payload = {
    schema: 'medagentaudit.audit.v1',
    auditor: { id: auditorId.value, name },
    exportedAt: new Date().toISOString(),
    annotations: loadAuditMap(auditorId.value),
  }
  downloadJson(`${name}_audit.json`, payload)
}
</script>

<template>
  <div class="grid gap-4 lg:grid-cols-[360px_1fr]">
    <AppCard class="p-4">
      <div class="space-y-3">
        <div>
          <div class="text-sm font-semibold text-slate-900">Auditor ID</div>
          <div class="mt-2">
            <AppSelect
              v-model="auditorIdText"
              :options="[
                { value: '1', label: 'Auditor #1' },
                { value: '2', label: 'Auditor #2' },
                { value: '3', label: 'Auditor #3' },
                { value: '4', label: 'Auditor #4' },
                { value: '5', label: 'Auditor #5' },
                { value: '6', label: 'Auditor #6' },
              ]"
            />
          </div>
          <div class="mt-1 text-xs text-slate-600">
            Deterministic assignment: each case is reviewed by exactly 3 auditors.
          </div>
        </div>

        <div class="flex items-center justify-between gap-2">
          <div class="text-xs text-slate-600">
            Done: <span class="font-semibold text-slate-900">{{ doneCount }}</span> / {{ assignedItems.length }}
          </div>
          <AppButton variant="secondary" :disabled="!auditorId" @click="exportJson">Export JSON</AppButton>
        </div>

        <div>
          <AppInput v-model="search" placeholder="Search auditId / caseId / taxonomy key ..." />
        </div>

        <div class="max-h-[70vh] overflow-auto pr-1">
          <div v-if="!auditorId" class="rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-600">
            Enter an auditor ID to load assigned items.
          </div>

          <div v-else class="space-y-2">
            <button
              v-for="it in filteredItems"
              :key="it.auditId"
              type="button"
              class="w-full rounded-xl border p-3 text-left text-sm transition"
              :class="
                it.auditId === activeAuditId ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-white hover:bg-slate-50'
              "
              @click="activeAuditId = it.auditId"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="truncate font-medium text-slate-900">{{ it.auditId }}</div>
                <div
                  class="shrink-0 rounded-md px-2 py-0.5 text-xs"
                  :class="annotations[it.auditId] ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-600'"
                >
                  {{ annotations[it.auditId] ? 'Done' : 'Todo' }}
                </div>
              </div>
              <div class="mt-1 text-xs text-slate-600">Case {{ it.caseId }} · {{ it.taxonomyKey }}</div>
            </button>
          </div>
        </div>
      </div>
    </AppCard>

    <div v-if="activeItem && activeCase" class="space-y-4">
      <AppCard class="p-5">
        <div class="flex flex-wrap items-start justify-between gap-2">
          <div>
            <div class="text-xs text-slate-600">{{ activeCase.dataset }} · {{ activeCase.framework }} · {{ activeCase.modality }}</div>
            <div class="mt-1 text-lg font-semibold text-slate-900">Audit {{ activeItem.auditId }}</div>
          </div>
          <div class="text-xs text-slate-600">Auto-saved on every change.</div>
        </div>

        <div class="mt-4 space-y-4">
          <div>
            <div class="text-sm font-semibold text-slate-900">Question</div>
            <div class="mt-2 whitespace-pre-wrap rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-800">
              {{ activeCase.question }}
            </div>
          </div>

          <div v-if="activeCase.modality === 'vqa' && activeCase.image?.path">
            <div class="text-sm font-semibold text-slate-900">Image</div>
            <img
              class="mt-2 max-h-[360px] w-auto rounded-xl border border-slate-200 bg-white"
              :src="activeCase.image.path"
              :alt="activeCase.image.alt ?? 'VQA image'"
            />
          </div>

          <div>
            <div class="text-sm font-semibold text-slate-900">Failure mode</div>
            <div class="mt-2 rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-800">
              <div class="flex flex-wrap items-center gap-2">
                <span class="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">
                  {{ activeItem.taxonomyKey }}
                </span>
                <span class="font-medium text-slate-900">{{ activeTaxonomy?.title ?? '' }}</span>
              </div>
              <div class="mt-2 text-xs text-slate-600">{{ activeTaxonomy?.short ?? '' }}</div>
            </div>
          </div>

          <div>
            <div class="text-sm font-semibold text-slate-900">Context (mode-level)</div>
            <div class="mt-2 whitespace-pre-wrap rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-800">
              {{ activeItem.context }}
            </div>
          </div>
        </div>
      </AppCard>

      <AppCard class="p-5">
        <div class="space-y-4">
          <div class="text-sm font-semibold text-slate-900">Verdict</div>
          <div class="flex flex-wrap gap-2">
            <button
              type="button"
              class="rounded-lg border px-3 py-2 text-sm font-medium transition"
              :class="verdict === 'yes' ? 'border-emerald-300 bg-emerald-50 text-emerald-800' : 'border-slate-200 bg-white hover:bg-slate-50'"
              @click="verdict = 'yes'"
            >
              Yes, the failure exists
            </button>
            <button
              type="button"
              class="rounded-lg border px-3 py-2 text-sm font-medium transition"
              :class="verdict === 'no' ? 'border-slate-400 bg-slate-100 text-slate-900' : 'border-slate-200 bg-white hover:bg-slate-50'"
              @click="verdict = 'no'"
            >
              No, not present
            </button>
          </div>

          <div class="flex flex-wrap items-center justify-between gap-3">
            <div class="text-xs text-slate-600">
              Local storage key: <span class="font-mono">medagentaudit:audit:auditor:{{ auditorId ?? '...' }}</span>
            </div>
            <div class="text-xs text-slate-600">Export from top-right when finished.</div>
          </div>
        </div>
      </AppCard>
    </div>

    <AppCard v-else class="p-5">
      <div class="text-sm text-slate-600">
        No assigned audit items. Check `src/data/audit/cases.ts` and your auditor ID.
      </div>
    </AppCard>
  </div>
</template>
