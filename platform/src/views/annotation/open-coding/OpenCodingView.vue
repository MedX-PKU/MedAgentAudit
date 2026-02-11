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
import type { OpenCodingCase, OpenCodingAnnotation } from '../../../domain/types'
import { copyToClipboard } from '../../../lib/clipboard'
import { downloadJson } from '../../../lib/download'
import { loadOpenCodingCases } from '../../../data/open-coding/cases'
import { loadOpenCodingMap, saveOpenCoding } from './openCodingStorage'

const annotatorName = ref('Annotator_1')
const search = ref('')
const activeCaseId = ref<string | null>(null)

const annotations = ref<Record<string, OpenCodingAnnotation>>({})
const cases = ref<OpenCodingCase[]>([])
const casesLoaded = ref(false)

const loadCases = async () => {
  try {
    cases.value = await loadOpenCodingCases()
  } catch (error) {
    console.error(error)
    cases.value = []
  } finally {
    casesLoaded.value = true
  }
}

watch(
  annotatorName,
  (name) => {
    if (!name.trim()) {
      annotations.value = {}
      activeCaseId.value = null
      return
    }
    annotations.value = loadOpenCodingMap(name.trim())
    if (!activeCaseId.value && cases.value.length > 0) activeCaseId.value = cases.value[0]!.caseId
  },
  { immediate: true },
)

watch(
  cases,
  (list) => {
    if (!activeCaseId.value && annotatorName.value.trim() && list.length > 0) {
      activeCaseId.value = list[0]!.caseId
    }
  },
  { deep: true },
)

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

type ParsedOutput = { answer?: string; explanation?: string }
type Opinion = { agent_id: string; specialty?: string; log?: { parsed_output?: ParsedOutput } }
type Review = {
  agent_id: string
  specialty?: string
  log?: { parsed_output?: { agree?: boolean; answer?: string; reason?: string; explanation?: string } }
}
type Round = {
  round: number
  opinions?: Opinion[]
  synthesis?: { parsed_output?: ParsedOutput }
  reviews?: Review[]
  decision?: { parsed_output?: ParsedOutput }
}

const collaborationRounds = computed<Round[]>(() => {
  const log = activeCase.value?.collaborationLog as { case_history?: { rounds?: Round[] } } | undefined
  const rounds = log?.case_history?.rounds
  return Array.isArray(rounds) ? rounds : []
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
const isAllDone = computed(() => cases.value.length > 0 && doneCount.value >= cases.value.length)
const completionToastShown = ref(false)

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

loadCases()

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

watch(isAllDone, (done) => {
  if (!done || completionToastShown.value) return
  toast.value = { show: true, message: 'All cases labeled. Please export your JSON.' }
  completionToastShown.value = true
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 2000)
})

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
    <div class="group fixed left-0 top-0 z-40 flex h-full items-center">
      <div class="pointer-events-none h-full w-3 bg-slate-900/5 transition group-hover:bg-slate-900/10"></div>
      <div
        class="pointer-events-auto ml-2 w-[320px] -translate-x-full rounded-2xl border border-slate-200 bg-white p-4 shadow-xl transition group-hover:translate-x-0"
      >
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

          <ProgressBar :done="doneCount" :total="cases.length" />

          <div class="flex flex-wrap gap-2">
            <AppButton variant="secondary" :disabled="!annotatorName.trim()" @click="activeCaseId = nextTodoCaseId">
              Next todo
            </AppButton>
            <AppButton v-if="isAllDone" variant="secondary" :disabled="!annotatorName.trim()" @click="exportJson">
              Export JSON
            </AppButton>
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
      </div>
    </div>

    <div class="grid gap-4 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)_minmax(0,420px)]">
      <div>
        <AppCard v-if="activeCase" class="max-h-[78vh] overflow-auto p-5">
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

        <AppCard v-else class="p-5">
          <div v-if="casesLoaded" class="text-sm text-slate-600">
            No open-coding cases found in `public/data/open-coding/`.
          </div>
          <div v-else class="text-sm text-slate-600">Loading open-coding cases...</div>
        </AppCard>
      </div>

      <div>
        <AppCard v-if="activeCase" class="max-h-[78vh] overflow-auto p-5">
          <div class="flex items-center justify-between gap-3">
            <div class="text-sm font-semibold text-slate-900">Collaboration log</div>
            <AppIconButton title="Copy log" variant="secondary" @click="copyLog">
              <span class="font-mono text-xs">Copy</span>
            </AppIconButton>
          </div>

          <div v-if="collaborationRounds.length" class="mt-4 space-y-4">
            <AppCard
              v-for="round in collaborationRounds"
              :key="round.round"
              class="space-y-3 border border-slate-200 bg-white p-4"
            >
              <div class="text-xs font-medium text-slate-500">Round {{ round.round }}</div>

              <div v-if="round.opinions?.length" class="space-y-2">
                <div class="text-sm font-semibold text-slate-900">Opinions</div>
                <div v-for="op in round.opinions" :key="`${round.round}-${op.agent_id}`" class="space-y-1">
                  <div class="text-xs font-medium text-slate-600">
                    {{ op.agent_id }} <span v-if="op.specialty">· {{ op.specialty }}</span>
                  </div>
                  <div class="text-xs text-slate-700">Answer: {{ op.log?.parsed_output?.answer ?? '-' }}</div>
                  <div v-if="op.log?.parsed_output?.explanation" class="text-xs text-slate-600 whitespace-pre-wrap">
                    {{ op.log.parsed_output.explanation }}
                  </div>
                </div>
              </div>

              <div v-if="round.synthesis?.parsed_output" class="space-y-1">
                <div class="text-sm font-semibold text-slate-900">Synthesis</div>
                <div class="text-xs text-slate-700">
                  Answer: {{ round.synthesis.parsed_output.answer ?? '-' }}
                </div>
                <div v-if="round.synthesis.parsed_output.explanation" class="text-xs text-slate-600 whitespace-pre-wrap">
                  {{ round.synthesis.parsed_output.explanation }}
                </div>
              </div>

              <div v-if="round.reviews?.length" class="space-y-2">
                <div class="text-sm font-semibold text-slate-900">Reviews</div>
                <div v-for="review in round.reviews" :key="`${round.round}-${review.agent_id}`" class="space-y-1">
                  <div class="text-xs font-medium text-slate-600">
                    {{ review.agent_id }} <span v-if="review.specialty">· {{ review.specialty }}</span>
                  </div>
                  <div class="text-xs text-slate-700">
                    Agree: {{ review.log?.parsed_output?.agree === undefined ? '-' : review.log.parsed_output.agree ? 'Yes' : 'No' }}
                  </div>
                  <div v-if="review.log?.parsed_output?.answer" class="text-xs text-slate-700">
                    Answer: {{ review.log.parsed_output.answer }}
                  </div>
                  <div v-if="review.log?.parsed_output?.reason" class="text-xs text-slate-600 whitespace-pre-wrap">
                    {{ review.log.parsed_output.reason }}
                  </div>
                  <div v-if="review.log?.parsed_output?.explanation" class="text-xs text-slate-600 whitespace-pre-wrap">
                    {{ review.log.parsed_output.explanation }}
                  </div>
                </div>
              </div>

              <div v-if="round.decision?.parsed_output" class="space-y-1">
                <div class="text-sm font-semibold text-slate-900">Decision</div>
                <div class="text-xs text-slate-700">
                  Answer: {{ round.decision.parsed_output.answer ?? '-' }}
                </div>
                <div v-if="round.decision.parsed_output.explanation" class="text-xs text-slate-600 whitespace-pre-wrap">
                  {{ round.decision.parsed_output.explanation }}
                </div>
              </div>
            </AppCard>
          </div>

          <div v-else class="mt-3 text-xs text-slate-600">
            No structured collaboration log found.
          </div>

        </AppCard>
      </div>

      <div>
        <AppCard v-if="activeCase" class="max-h-[78vh] overflow-auto p-5">
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
      </div>
    </div>

    <AppToast :show="toast.show" :message="toast.message" />
  </div>
</template>
