<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { TAXONOMY, type TaxonomyKey } from '../../../domain/taxonomy'
import AppButton from '../../../components/ui/AppButton.vue'
import AppCard from '../../../components/ui/AppCard.vue'
import AppSelect from '../../../components/ui/AppSelect.vue'
import ProgressBar from '../../../components/annotation/ProgressBar.vue'
import TwoPane from '../../../components/layout/TwoPane.vue'
import AppToast from '../../../components/ui/AppToast.vue'
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
const toast = ref<{ show: boolean; message: string }>({ show: false, message: '' })

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
const isAllDone = computed(() => assignedItems.value.length > 0 && doneCount.value >= assignedItems.value.length)
const completionHintShown = ref(false)

const nextTodoAuditId = computed(() => {
  const list = filteredItems.value
  const idx = list.findIndex((it) => it.auditId === activeAuditId.value)
  const start = idx >= 0 ? idx : 0
  for (let offset = 0; offset < list.length; offset++) {
    const it = list[(start + offset) % list.length]!
    if (!annotations.value[it.auditId]) return it.auditId
  }
  return list[0]?.auditId ?? null
})

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

watch(isAllDone, (done) => {
  if (!done || completionHintShown.value) return
  completionHintShown.value = true
  toast.value = { show: true, message: 'All items reviewed. Please export your JSON.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 2000)
})

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
  <TwoPane>
    <template #left>
      <AppCard class="p-4">
        <div class="space-y-3">
          <div>
            <div class="text-sm font-semibold text-slate-900">Auditor</div>
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

          <div v-if="auditorId" class="space-y-3">
            <ProgressBar :done="doneCount" :total="assignedItems.length" />

            <div class="flex flex-wrap gap-2">
              <AppButton variant="secondary" :disabled="!auditorId" @click="activeAuditId = nextTodoAuditId">
                Next todo
              </AppButton>
              <AppButton v-if="isAllDone" variant="secondary" :disabled="!auditorId" @click="exportJson">
                Export JSON
              </AppButton>
            </div>

            <input
              v-model="search"
              placeholder="Search auditId / caseId / taxonomy key ..."
              class="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm outline-none ring-blue-500/20 placeholder:text-slate-400 focus:ring-2"
            />
          </div>
        </div>

        <div class="mt-4 max-h-[60vh] overflow-auto pr-1">
          <div v-if="!auditorId" class="rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-600">
            Select an auditor ID to load assigned items.
          </div>

          <div v-else class="space-y-2">
            <button
              v-for="it in filteredItems"
              :key="it.auditId"
              type="button"
              class="w-full rounded-xl border p-3 text-left text-sm transition"
              :class="it.auditId === activeAuditId ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-white hover:bg-slate-50'"
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
      </AppCard>
    </template>

    <template #right>
      <AppCard v-if="activeItem && activeCase" class="p-5">
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <div class="text-sm font-semibold text-slate-900">Verdict</div>
            <div class="text-xs text-slate-500">Always visible</div>
          </div>

          <div class="space-y-2">
            <button
              type="button"
              class="w-full rounded-xl border px-3 py-3 text-left text-sm font-medium transition"
              :class="verdict === 'yes' ? 'border-emerald-300 bg-emerald-50 text-emerald-900' : 'border-slate-200 bg-white hover:bg-slate-50'"
              @click="verdict = 'yes'"
            >
              <div class="font-semibold">Yes</div>
              <div class="mt-1 text-xs text-slate-600">The failure mode is present in this context.</div>
            </button>
            <button
              type="button"
              class="w-full rounded-xl border px-3 py-3 text-left text-sm font-medium transition"
              :class="verdict === 'no' ? 'border-slate-400 bg-slate-100 text-slate-900' : 'border-slate-200 bg-white hover:bg-slate-50'"
              @click="verdict = 'no'"
            >
              <div class="font-semibold">No</div>
              <div class="mt-1 text-xs text-slate-600">The failure mode is not supported by this context.</div>
            </button>
          </div>

          <div class="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
            Auto-saved. Export from the left panel when finished.
          </div>
        </div>
      </AppCard>
    </template>

    <template #main>
      <div v-if="activeItem && activeCase" class="space-y-4">
        <AppCard class="p-5">
          <div class="flex flex-wrap items-start justify-between gap-2">
            <div>
              <div class="text-xs text-slate-600">{{ activeCase.dataset }} · {{ activeCase.framework }} · {{ activeCase.modality }}</div>
              <div class="mt-1 text-lg font-semibold text-slate-900">Audit {{ activeItem.auditId }}</div>
            </div>
            <div class="text-xs text-slate-600">
              Auto-saved
              <span class="font-mono">medagentaudit:audit:auditor:{{ auditorId }}</span>
            </div>
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
      </div>

      <AppCard v-else class="p-5">
        <div class="text-sm text-slate-600">No assigned audit items. Select an auditor and item.</div>
      </AppCard>
    </template>
  </TwoPane>

  <AppToast :show="toast.show" :message="toast.message" />
</template>
