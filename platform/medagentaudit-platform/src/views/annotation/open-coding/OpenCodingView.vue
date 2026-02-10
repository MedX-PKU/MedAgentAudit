<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import TaxonomyChecklist from '../../../components/annotation/TaxonomyChecklist.vue'
import type { TaxonomyKey } from '../../../domain/taxonomy'
import AppButton from '../../../components/ui/AppButton.vue'
import AppCard from '../../../components/ui/AppCard.vue'
import AppIconButton from '../../../components/ui/AppIconButton.vue'
import AppInput from '../../../components/ui/AppInput.vue'
import AppSelect from '../../../components/ui/AppSelect.vue'
import AppTextarea from '../../../components/ui/AppTextarea.vue'
import AppToast from '../../../components/ui/AppToast.vue'
import ProgressBar from '../../../components/annotation/ProgressBar.vue'
import TwoPane from '../../../components/layout/TwoPane.vue'
import type { OpenCodingCase, OpenCodingAnnotation } from '../../../domain/types'
import { copyToClipboard } from '../../../lib/clipboard'
import { downloadJson } from '../../../lib/download'
import { OPEN_CODING_CASES } from '../../../data/open-coding/cases'
import { loadOpenCodingMap, saveOpenCoding } from './openCodingStorage'

const annotatorName = ref('Annotator_1')
const search = ref('')
const activeCaseId = ref<string | null>(null)

const annotations = ref<Record<string, OpenCodingAnnotation>>({})

watch(
  annotatorName,
  (name) => {
    if (!name.trim()) {
      annotations.value = {}
      activeCaseId.value = null
      return
    }
    annotations.value = loadOpenCodingMap(name.trim())
    if (!activeCaseId.value && OPEN_CODING_CASES.length > 0) activeCaseId.value = OPEN_CODING_CASES[0]!.caseId
  },
  { immediate: true },
)

const cases = computed(() => OPEN_CODING_CASES)

const filteredCases = computed(() => {
  const q = search.value.trim().toLowerCase()
  if (!q) return cases.value
  return cases.value.filter((c) => {
    const hay = [c.caseId, c.dataset, c.framework, c.question].join(' ').toLowerCase()
    return hay.includes(q)
  })
})

const activeCase = computed<OpenCodingCase | null>(() => {
  if (!activeCaseId.value) return null
  return cases.value.find((c) => c.caseId === activeCaseId.value) ?? null
})

const activeAnnotation = computed<OpenCodingAnnotation | null>(() => {
  if (!activeCase.value) return null
  return annotations.value[activeCase.value.caseId] ?? null
})

const taxonomy = ref<TaxonomyKey[]>([])
const novelFailureMode = ref('')
const toast = ref<{ show: boolean; message: string }>({ show: false, message: '' })

watch(
  activeAnnotation,
  (a) => {
    taxonomy.value = a?.taxonomy ?? []
    novelFailureMode.value = a?.novelFailureMode ?? ''
  },
  { immediate: true },
)

const doneCount = computed(() => Object.keys(annotations.value).length)

const nextTodoCaseId = computed(() => {
  const list = filteredCases.value
  const idx = list.findIndex((c) => c.caseId === activeCaseId.value)
  const start = idx >= 0 ? idx : 0
  for (let offset = 0; offset < list.length; offset++) {
    const c = list[(start + offset) % list.length]!
    if (!annotations.value[c.caseId]) return c.caseId
  }
  return list[0]?.caseId ?? null
})

const persistActive = () => {
  if (!annotatorName.value.trim() || !activeCase.value) return
  const annotation: OpenCodingAnnotation = {
    caseId: activeCase.value.caseId,
    taxonomy: taxonomy.value,
    novelFailureMode: novelFailureMode.value.trim() || undefined,
    updatedAt: new Date().toISOString(),
  }
  saveOpenCoding(annotatorName.value.trim(), annotation)
  annotations.value = loadOpenCodingMap(annotatorName.value.trim())
}

watch([taxonomy, novelFailureMode], persistActive, { deep: true })

const exportJson = () => {
  const name = annotatorName.value.trim()
  if (!name) return
  const payload = {
    schema: 'medagentaudit.open_coding.v1',
    annotator: { name },
    exportedAt: new Date().toISOString(),
    annotations: loadOpenCodingMap(name),
  }
  downloadJson(`${name}_opencoding.json`, payload)
}

const copyLog = async () => {
  if (!activeCase.value) return
  await copyToClipboard(JSON.stringify(activeCase.value.collaborationLog, null, 2))
  toast.value = { show: true, message: 'Copied collaboration log.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 1200)
}
</script>

<template>
  <div class="space-y-4">
    <TwoPane>
      <template #left>
        <AppCard class="p-4">
          <div class="space-y-3">
            <div>
              <div class="text-sm font-semibold text-slate-900">Annotator</div>
              <div class="mt-2">
                <AppSelect
                  v-model="annotatorName"
                  :options="[
                    { value: 'Annotator_1', label: 'Annotator #1' },
                    { value: 'Annotator_2', label: 'Annotator #2' },
                  ]"
                />
              </div>
            </div>

            <AppInput v-model="search" placeholder="Search caseId / dataset / framework ..." />

            <ProgressBar :done="doneCount" :total="OPEN_CODING_CASES.length" />

            <div class="flex flex-wrap gap-2">
              <AppButton variant="secondary" :disabled="!annotatorName.trim()" @click="activeCaseId = nextTodoCaseId">
                Next todo
              </AppButton>
              <AppButton variant="secondary" :disabled="!annotatorName.trim()" @click="exportJson">Export JSON</AppButton>
            </div>
          </div>

          <div class="mt-4 max-h-[60vh] overflow-auto pr-1">
            <div class="space-y-2">
              <button
                v-for="c in filteredCases"
                :key="c.caseId"
                type="button"
                class="w-full rounded-xl border p-3 text-left text-sm transition"
                :class="c.caseId === activeCaseId ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-white hover:bg-slate-50'"
                @click="activeCaseId = c.caseId"
              >
                <div class="flex items-center justify-between gap-2">
                  <div class="truncate font-medium text-slate-900">{{ c.caseId }}</div>
                  <div
                    class="shrink-0 rounded-md px-2 py-0.5 text-xs"
                    :class="annotations[c.caseId] ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-600'"
                  >
                    {{ annotations[c.caseId] ? 'Done' : 'Todo' }}
                  </div>
                </div>
                <div class="mt-1 text-xs text-slate-600">{{ c.dataset }} · {{ c.framework }} · {{ c.modality }}</div>
              </button>
            </div>
          </div>
        </AppCard>
      </template>

      <template #right>
        <AppCard v-if="activeCase" class="p-5">
          <div class="space-y-4">
            <div class="flex items-center justify-between">
              <div class="text-sm font-semibold text-slate-900">Labeling</div>
              <div class="text-xs text-slate-500">Always visible</div>
            </div>
            <TaxonomyChecklist v-model="taxonomy" />
            <div>
              <div class="text-sm font-semibold text-slate-900">Novel failure mode (optional)</div>
              <div class="mt-2">
                <AppTextarea v-model="novelFailureMode" :rows="4" placeholder="Describe a new failure mode if needed..." />
              </div>
            </div>
          </div>
        </AppCard>
      </template>

      <template #main>
        <div v-if="activeCase" class="space-y-4">
          <AppCard class="p-5">
            <div class="flex items-start justify-between gap-3">
              <div>
                <div class="text-xs text-slate-600">{{ activeCase.dataset }} · {{ activeCase.framework }} · {{ activeCase.modality }}</div>
                <div class="mt-1 text-lg font-semibold text-slate-900">Case {{ activeCase.caseId }}</div>
              </div>
              <div class="text-xs text-slate-600">
                Auto-saved
                <span class="font-mono">medagentaudit:open-coding:{{ annotatorName.trim() }}</span>
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

              <div v-if="activeCase.options?.length">
                <div class="text-sm font-semibold text-slate-900">Options</div>
                <ul class="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
                  <li v-for="(opt, idx) in activeCase.options" :key="idx">{{ opt }}</li>
                </ul>
              </div>
            </div>
          </AppCard>

          <AppCard class="p-5">
            <div class="flex items-center justify-between gap-3">
              <div class="text-sm font-semibold text-slate-900">Collaboration log (full)</div>
              <AppIconButton title="Copy log" variant="secondary" @click="copyLog">
                <span class="font-mono text-xs">Copy</span>
              </AppIconButton>
            </div>
            <pre class="mt-3 max-h-[75vh] overflow-auto rounded-lg bg-slate-950 p-3 text-xs text-slate-100">{{
              JSON.stringify(activeCase.collaborationLog, null, 2)
            }}</pre>
          </AppCard>
        </div>

        <AppCard v-else class="p-5">
          <div class="text-sm text-slate-600">No open-coding cases found in `src/data/open-coding/cases.ts`.</div>
        </AppCard>
      </template>
    </TwoPane>

    <AppToast :show="toast.show" :message="toast.message" />
  </div>
</template>
