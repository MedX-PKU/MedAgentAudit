<script setup lang="ts">
import { computed, inject, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import MarkdownIt from 'markdown-it'

import { ANNOTATION_DRAWER_KEY, type AnnotationDrawerContext } from '../../../components/layout/annotationDrawer'

import type { TaxonomyKey } from '../../../domain/taxonomy'
import AppButton from '../../../components/ui/AppButton.vue'
import AppCard from '../../../components/ui/AppCard.vue'
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
const activeCaseId = ref<string | null>(null)
const isDrawerOpen = ref(false)
const isInstructionPopoverOpen = ref(false)

const annotationDrawer = inject<AnnotationDrawerContext | null>(ANNOTATION_DRAWER_KEY, null)

const OPEN_CODING_INSTRUCTION_TEXT =
  "Please conduct a comprehensive analysis of the multi-agent collaboration process for this case, utilizing the full case context and collaboration history provided.\n\nYour task is to identify occurrences of the 10 specific failure modes listed in the taxonomy.\n\nFor each failure mode observed, please select (check) the corresponding checkbox.\n\nIf a failure mode is not present, leave it unchecked (do not take any action).\n\nShould you encounter any other collaboration issues not covered by these 10 categories, please describe them in the 'Novel failure mode' text box."

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
    const list = cases.value
    const currentValid = list.some((c) => c.caseId === activeCaseId.value)
    if ((!activeCaseId.value || !currentValid) && list.length > 0) {
      activeCaseId.value = list[0]!.caseId
    }
  },
  { immediate: true },
)

// Sync from layout (hamburger/rail) -> local drawer state.
watch(
  () => annotationDrawer?.isOpen.value,
  (open) => {
    if (open == null) return
    isDrawerOpen.value = open
  },
  { immediate: true },
)

watch(
  cases,
  (list) => {
    if (!annotatorName.value.trim() || list.length === 0) return
    const currentValid = list.some((c) => c.caseId === activeCaseId.value)
    if ((!activeCaseId.value || !currentValid) && list.length > 0) {
      activeCaseId.value = list[0]!.caseId
    }
  },
  { deep: true },
)

watch(activeCaseId, () => {
  // Keep drawer state when switching cases; close only popovers.
  isInstructionPopoverOpen.value = false
})


const activeCase = computed<OpenCodingCase | null>(() => {
  if (!activeCaseId.value) return null
  return cases.value.find((c) => c.caseId === activeCaseId.value) ?? null
})

const activeAnnotation = computed<OpenCodingAnnotation | null>(() => {
  if (!activeCase.value) return null
  const key = `case_${activeCase.value.seq ?? 'unknown'}_${activeCase.value.caseId}`
  return annotations.value[key] ?? null
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

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
})

const collaborationMarkdownHtml = computed(() => {
  const log = activeCase.value?.collaborationLog as { raw?: string } | undefined
  const raw = log?.raw
  if (!raw) return ''
  return md.render(raw)
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

watch(
  taxonomy,
  (list) => {
    if (list.includes('0.0.0')) {
      const onlyNone = list.length === 1 ? list : (['0.0.0'] as TaxonomyKey[])
      if (onlyNone.length !== list.length) taxonomy.value = onlyNone
      novelFailureMode.value = ''
    }
  },
  { deep: true },
)

const doneCount = computed(() => Object.keys(annotations.value).length)
const isAllDone = computed(() => cases.value.length > 0 && doneCount.value >= cases.value.length)
const completionToastShown = ref(false)

const nextTodoCaseId = computed(() => {
  const list = cases.value
  const idx = list.findIndex((c) => c.caseId === activeCaseId.value)
  const start = idx >= 0 ? idx : 0
  for (let offset = 0; offset < list.length; offset++) {
    const c = list[(start + offset) % list.length]!
    const key = `case_${c.seq ?? 'unknown'}_${c.caseId}`
    if (!annotations.value[key]) return c.caseId
  }
  return list[0]?.caseId ?? null
})

const goNext = () => {
  const hasTaxonomy = taxonomy.value.length > 0
  const hasNovel = novelFailureMode.value.trim().length > 0
  if (!hasTaxonomy && !hasNovel) {
    toast.value = { show: true, message: 'Please select at least one taxonomy item or enter a novel failure mode before moving on.' }
    window.setTimeout(() => {
      toast.value = { show: false, message: '' }
    }, 1500)
    return
  }
  activeCaseId.value = nextTodoCaseId.value
}

loadCases()

const persistActive = () => {
  if (!annotatorName.value.trim() || !activeCase.value) return
  const annotation: OpenCodingAnnotation = {
    caseId: activeCase.value.caseId,
    seq: activeCase.value.seq,
    taxonomy: taxonomy.value,
    novelFailureMode: novelFailureMode.value.trim() || undefined,
    updatedAt: new Date().toISOString(),
  }
  saveOpenCoding(annotatorName.value.trim(), annotation)
  annotations.value = loadOpenCodingMap(annotatorName.value.trim())
}

watch(
  [taxonomy, novelFailureMode],
  () => {
    const a = activeAnnotation.value
    const currentTaxonomy = a?.taxonomy ?? []
    const currentNovel = a?.novelFailureMode ?? ''
    const nextNovel = novelFailureMode.value.trim() || ''
    const sortedTaxonomy = [...taxonomy.value].sort()
    const sortedCurrent = [...currentTaxonomy].sort()
    const sameTaxonomy =
      sortedTaxonomy.length === sortedCurrent.length && sortedTaxonomy.every((k, i) => k === sortedCurrent[i])
    if (sameTaxonomy && nextNovel === currentNovel) return
    persistActive()
  },
  { deep: true },
)

type FailureModePopoverState = {
  open: boolean
  x: number
  y: number
  payload: { definition: string; instruction: string }
}

const failureModePopover = ref<FailureModePopoverState>({
  open: false,
  x: 0,
  y: 0,
  payload: { definition: '', instruction: '' },
})

const showFailureModePopover = (
  event: MouseEvent,
  item: { definition?: string; human_eval_instruction?: string; humanEvalInstruction?: string },
) => {
  const target = event.currentTarget as HTMLElement | null
  if (!target) return
  const rect = target.getBoundingClientRect()
  const popoverWidth = 520
  const margin = 12

  const left = Math.min(rect.left + rect.width / 2, window.innerWidth - margin)
  const top = rect.bottom + 8
  const translateX = Math.min(0, window.innerWidth - (rect.left + popoverWidth) - margin)

  failureModePopover.value = {
    open: true,
    x: left + translateX,
    y: Math.min(top, window.innerHeight - margin),
    payload: {
      definition: item.definition ?? '-',
      instruction: item.human_eval_instruction ?? item.humanEvalInstruction ?? '-',
    },
  }
}

const hideFailureModePopover = () => {
  if (!failureModePopover.value.open) return
  failureModePopover.value = { ...failureModePopover.value, open: false }
}

const toggle = (key: TaxonomyKey) => {
  const set = new Set(taxonomy.value)
  const noneKey: TaxonomyKey = '0.0.0'
  if (key === noneKey) {
    taxonomy.value = set.has(noneKey) ? [] : [noneKey]
    return
  }
  if (set.has(key)) set.delete(key)
  else set.add(key)
  if (set.has(noneKey)) set.delete(noneKey)
  taxonomy.value = Array.from(set)
}

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

  const rawAnnotations = loadOpenCodingMap(name)
  const enrichedAnnotations = Object.fromEntries(
    Object.entries(rawAnnotations).map(([key, ann]) => {
      const c = cases.value.find((x) => x.caseId === ann.caseId && (ann.seq ? x.seq === ann.seq : true))
      return [
        key,
        {
          ...ann,
          seq: ann.seq ?? c?.seq,
          dataset: ann.dataset ?? c?.dataset,
          mas: ann.mas ?? c?.framework,
          llm: ann.llm ?? c?.llm,
        },
      ]
    }),
  )

  const payload = {
    schema: 'medagentaudit.open_coding.v1',
    annotator: { name },
    exportedAt: new Date().toISOString(),
    annotations: enrichedAnnotations,
  }
  downloadJson(`${name}_opencoding.json`, payload)
}

const copyLog = async () => {
  const log = activeCase.value?.collaborationLog as { raw?: string } | undefined
  const raw = log?.raw
  if (!raw) return
  await copyToClipboard(raw)
  toast.value = { show: true, message: 'Copied Collaboration Text.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 1200)
}

const copyQuestion = async () => {
  if (!activeCase.value) return
  const questionText = activeCase.value.question ?? ''
  const questionTypeText = activeCase.value.questionType ?? ''
  const optionsText = (activeCase.value.options ?? []).join('\n')
  const payload = `Question: ${questionText}\nQuestion Type: ${questionTypeText}\nOptions: ${optionsText}`
  await copyToClipboard(payload)
  toast.value = { show: true, message: 'Copied Question and Options.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 1200)
}

const copyInstruction = async () => {
  await copyToClipboard(OPEN_CODING_INSTRUCTION_TEXT)
  toast.value = { show: true, message: 'Copied Instruction Text.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 1200)
}

const toggleInstructionPopover = () => {
  isInstructionPopoverOpen.value = !isInstructionPopoverOpen.value
}

const onDocumentClick = (event: MouseEvent) => {
  // Close drawer + instruction popover when clicking outside.
  if (!isDrawerOpen.value && !isInstructionPopoverOpen.value) return
  const target = event.target as HTMLElement | null
  if (!target) return
  if (target.closest('[data-drawer]')) return
  if (target.closest('[data-instruction-popover]')) return
  annotationDrawer?.close?.()
  isInstructionPopoverOpen.value = false
}

onMounted(() => {
  document.addEventListener('click', onDocumentClick)
  window.addEventListener('scroll', hideFailureModePopover, true)
  window.addEventListener('resize', hideFailureModePopover)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', onDocumentClick)
  window.removeEventListener('scroll', hideFailureModePopover, true)
  window.removeEventListener('resize', hideFailureModePopover)
})
</script>

<template>
  <div class="space-y-4">
	    <div
        class="fixed left-0 top-3 z-40 w-[320px] max-h-[calc(100vh-1.5rem)] flex flex-col rounded-2xl border border-slate-200 bg-white p-4 shadow-xl transition"
        data-drawer
        :class="isDrawerOpen ? 'pointer-events-auto translate-x-0 opacity-100' : 'pointer-events-none -translate-x-full opacity-0'"
        :style="{ visibility: isDrawerOpen ? 'visible' : 'hidden' }"
      >
        <AppCard class="p-4 flex flex-col min-h-0">
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
            <div class="mt-1 text-xs text-slate-600">
              Select an annotator to load and label open-coding cases.
            </div>
          </div>

          <ProgressBar :done="doneCount" :total="cases.length" />

          <div class="flex flex-wrap gap-2">
            <AppButton variant="secondary" :disabled="!annotatorName.trim()" @click="activeCaseId = nextTodoCaseId">
              Next TODO
            </AppButton>
            <AppButton variant="secondary" :disabled="!annotatorName.trim()" @click="exportJson">
              Export JSON
            </AppButton>
          </div>
        </div>

        <div class="mt-4 max-h-[60vh] min-h-[200px] overflow-auto pr-1">
          <div class="space-y-2">
            <button
              v-for="c in cases"
              :key="c.caseId"
              type="button"
              class="w-full rounded-xl border p-3 text-left text-sm transition"
              :class="c.caseId === activeCaseId ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-white hover:bg-slate-50'"
              @click="activeCaseId = c.caseId"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="truncate font-medium text-slate-900">Case {{ c.seq }}: {{ c.caseId }}</div>
                <div
                  class="shrink-0 rounded-md px-2 py-0.5 text-xs"
                  :class="annotations[`case_${c.seq ?? 'unknown'}_${c.caseId}`] ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-600'"
                >
                  {{ annotations[`case_${c.seq ?? 'unknown'}_${c.caseId}`] ? 'Done' : 'TODO' }}
                </div>
              </div>
              <div class="mt-1 text-xs text-slate-600">{{ c.dataset }} · {{ c.framework }} · {{ c.modality }}</div>
            </button>
          </div>
        </div>
      </AppCard>
      </div>

    <div class="grid gap-4 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)_minmax(0,520px)]">
      <div>
        <AppCard v-if="activeCase" class="max-h-[86vh] overflow-auto p-5">
          <div class="flex items-start justify-between gap-3">
            <div>
              <div class="mt-1 text-lg font-semibold text-slate-900">Case {{ activeCase.seq }}: {{ activeCase.caseId }}</div>
            </div>
          </div>

          <div class="mt-4 space-y-4">
            <div>
              <div class="flex items-center justify-between gap-3">
                <div class="text-sm font-semibold text-slate-900">Question</div>
                <AppButton variant="secondary" @click="copyQuestion">Copy Question & Options</AppButton>
              </div>
              <div class="mt-2 whitespace-pre-wrap rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-800">
                {{ activeCase.question }}
              </div>
              <div v-if="activeCase.questionType" class="mt-4 text-xs text-slate-600">
                Question Type: {{ activeCase.questionType }}
              </div>
            </div>

            <div v-if="activeCase.modality === 'vqa' && activeCase.image?.path">
              <div class="flex items-center justify-between gap-3">
                <div class="text-sm font-semibold text-slate-900">Image</div>
              </div>
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

            <div v-if="activeCase.collectionText">
              <div class="text-sm font-semibold text-slate-900">Collection Text</div>
              <div class="mt-2 whitespace-pre-wrap rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-800">
                {{ activeCase.collectionText }}
              </div>
            </div>

            <div v-if="activeCase.answer || activeCase.predictedAnswer" class="flex flex-wrap gap-2 text-sm">
              <div v-if="activeCase.answer" class="rounded-md bg-emerald-50 px-2 py-1 font-semibold text-emerald-800">
                Ground Truth: {{ activeCase.answer }}
              </div>
              <div
                v-if="activeCase.predictedAnswer"
                class="rounded-md bg-indigo-50 px-2 py-1 font-semibold text-indigo-800"
              >
                Predicted Answer: {{ activeCase.predictedAnswer }}
              </div>
              <div
                v-if="activeCase.answer && activeCase.predictedAnswer"
                class="rounded-md px-2 py-1 font-semibold text-white"
                :class="activeCase.answer === activeCase.predictedAnswer ? 'bg-emerald-600' : 'bg-rose-600'"
              >
                {{ activeCase.answer === activeCase.predictedAnswer ? 'Correct' : 'Incorrect' }}
              </div>
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
        <AppCard v-if="activeCase" class="max-h-[86vh] overflow-visible p-5">
          <div class="flex items-center justify-between gap-3">
            <div class="text-sm font-semibold text-slate-900">Collaboration log</div>
            <div class="flex items-center gap-2">
              <div class="relative inline-block" data-instruction-popover>
                <AppButton variant="secondary" @click="toggleInstructionPopover">
                  {{ isInstructionPopoverOpen ? 'Hide Instruction' : 'Show Instruction' }}
                </AppButton>
                <div
                  v-if="isInstructionPopoverOpen"
                  class="absolute left-0 top-full z-[99999] mt-2 w-[520px] rounded-xl border border-slate-200 bg-white p-3 text-xs text-slate-900 shadow-lg"
                >
                  <div class="mb-2 flex items-center justify-between gap-3">
                    <div class="text-sm font-semibold text-slate-900">Instruction Text</div>
                    <AppButton variant="secondary" @click="copyInstruction">Copy Instruction</AppButton>
                  </div>
                  <div class="whitespace-pre-wrap">{{ OPEN_CODING_INSTRUCTION_TEXT }}</div>
                </div>
              </div>
              <AppButton variant="secondary" @click="copyLog">Copy Collaboration</AppButton>
            </div>
          </div>

          <div
            v-if="collaborationMarkdownHtml"
            class="mt-3 max-h-[75vh] overflow-auto rounded-xl border border-slate-200 bg-white p-4"
          >
            <div class="prose prose-slate max-w-none text-sm" v-html="collaborationMarkdownHtml" />
          </div>

            <div v-else-if="collaborationRounds.length" class="mt-4 space-y-4">
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
        <AppCard v-if="activeCase" class="max-h-[86vh] overflow-auto p-5">
          <div class="space-y-4">
            <div class="flex items-center justify-between">
              <div class="text-sm font-semibold text-slate-900">Labeling</div>
            </div>
            <div class="space-y-2">
              <div
                v-for="(item, key) in activeCase.failureModeDefinitionMapping"
                :key="String(key)"
                class="relative z-0 flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2 hover:bg-slate-50"
              >
                <label class="flex cursor-pointer items-center">
                  <input
                    class="h-4 w-4 accent-blue-600"
                    type="checkbox"
                    :checked="taxonomy.includes(String(key) as TaxonomyKey)"
                    @change="toggle(String(key) as TaxonomyKey)"
                  />
                </label>

                <div class="min-w-0 flex-1 text-xs font-medium text-slate-800">
                  <span class="block whitespace-normal break-words">
                    {{ `${key}: ${item.name ?? '-'}` }}
                  </span>
                </div>

                <button
                  type="button"
                  class="shrink-0 rounded-full border border-blue-300 bg-blue-300 px-1.5 py-0.5 text-[10px] font-semibold leading-none text-white hover:bg-blue-700 hover:border-blue-700"
                  @mouseenter="showFailureModePopover($event, item)"
                  @mouseleave="hideFailureModePopover"
                >
                  i
                </button>
              </div>
            </div>
            <div class="space-y-3">
              <div class="relative z-0 flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2 hover:bg-slate-50">
                <label class="flex cursor-pointer items-center">
                  <input
                    class="h-4 w-4 accent-blue-600"
                    type="checkbox"
                    :checked="taxonomy.includes('0.0.0')"
                    @change="toggle('0.0.0')"
                  />
                </label>
                <div class="min-w-0 flex-1 text-xs font-medium text-slate-800">
                  <span class="block whitespace-normal break-words">0.0.0: No issues / No failure mode</span>
                </div>
              </div>

              <div class="grid gap-4">
                <div class="max-w-none">
                  <div class="text-sm font-semibold text-slate-900">Novel failure mode (optional)</div>
                  <div class="mt-2 flex items-end gap-4">
                    <AppTextarea
                      v-model="novelFailureMode"
                      :rows="2"
                      :placeholder="
                        taxonomy.includes('0.0.0')
                          ? 'Disabled when “No issues / No failure mode” is selected.'
                          : 'Describe a new failure mode if needed...'
                      "
                      :disabled="taxonomy.includes('0.0.0')"
                      class="h-[50px] w-1/2"
                    />
                    <AppButton
                      variant="secondary"
                      :disabled="!annotatorName.trim() || !nextTodoCaseId"
                      class="h-[50px] w-1/2 py-0"
                      @click="goNext"
                    >
                      Next TODO
                    </AppButton>
                  </div>
                </div>

                <div />
              </div>
            </div>
          </div>
        </AppCard>
      </div>
    </div>

    <div
      v-if="failureModePopover.open"
      class="pointer-events-none fixed z-[2147483647] w-[520px] whitespace-pre-wrap rounded-xl border border-slate-200 bg-white p-3 text-left text-xs text-slate-900 shadow-lg"
      :style="{ left: `${failureModePopover.x}px`, top: `${failureModePopover.y}px`, transform: 'translate(-50%, 0)' }"
    >
      <div class="font-semibold">Definition</div>
      <div class="mt-1 text-slate-800">{{ failureModePopover.payload.definition }}</div>
      <div class="mt-3 font-semibold">Evaluation Instruction</div>
      <div class="mt-1 text-slate-800">{{ failureModePopover.payload.instruction }}</div>
    </div>

    <AppToast :show="toast.show" :message="toast.message" />
  </div>
</template>
