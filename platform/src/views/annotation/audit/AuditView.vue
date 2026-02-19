<script setup lang="ts">
import { computed, inject, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import { FAILURE_MODE_DEFINITION_MAPPING } from '../../../domain/failureModes'
import AppButton from '../../../components/ui/AppButton.vue'
import AppCard from '../../../components/ui/AppCard.vue'
import AppSelect from '../../../components/ui/AppSelect.vue'
import ProgressBar from '../../../components/annotation/ProgressBar.vue'
import AppToast from '../../../components/ui/AppToast.vue'
import type { AuditAnnotation, AuditCase, AuditItem } from '../../../domain/types'
import { downloadJson } from '../../../lib/download'
import { copyToClipboard } from '../../../lib/clipboard'
import MarkdownIt from 'markdown-it'
import { loadAuditCases } from '../../../data/audit/cases'
import { assignedAuditCases, assignedAuditItems, type AuditorId } from './auditAssignment'
import { loadAuditMap, saveAudit } from './auditStorage'

import { ANNOTATION_DRAWER_KEY, type AnnotationDrawerContext } from '../../../components/layout/annotationDrawer'

type Verdict = 'yes' | 'no'

const auditorIdText = ref('1')
const AUDITOR_LABELS: Record<string, string> = {
  '1': '顾磊',
  '2': '桑浩然',
  '3': '王子翔',
  '4': '睢德昊',
  '5': '薛在宥',
  '6': 'Luna',
}

const auditorOptions = computed(() =>
  Object.entries(AUDITOR_LABELS).map(([value, label]) => ({
    value,
    label,
  })),
)

const auditorId = computed<AuditorId | null>(() => {
  const n = Number(auditorIdText.value)
  if (![1, 2, 3, 4, 5, 6].includes(n)) return null
  return n as AuditorId
})

const activeAuditId = ref<string | null>(null)
const verdict = ref<Verdict | null>(null)
const toast = ref<{ show: boolean; message: string }>({ show: false, message: '' })
const isDrawerOpen = ref(false)
const isInstructionPopoverOpen = ref(false)

const annotationDrawer = inject<AnnotationDrawerContext | null>(ANNOTATION_DRAWER_KEY, null)

const annotations = ref<Record<string, AuditAnnotation>>({})
const auditCases = ref<AuditCase[]>([])
const casesLoaded = ref(false)

const loadCases = async () => {
  try {
    auditCases.value = await loadAuditCases()
  } catch (error) {
    console.error(error)
    auditCases.value = []
  } finally {
    casesLoaded.value = true
  }
}

watch(
  auditorId,
  (id) => {
    if (!id) {
      annotations.value = {}
      activeAuditId.value = null
      return
    }
    annotations.value = loadAuditMap(id)
    const items = assignedAuditItems(id, auditCases.value)
    const currentValid = items.some((it) => it.auditId === activeAuditId.value)
    if ((!activeAuditId.value || !currentValid) && items.length > 0) {
      activeAuditId.value = items[0]!.auditId
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

const assignedItems = computed<AuditItem[]>(() => {
  if (!auditorId.value) return []
  return assignedAuditItems(auditorId.value, auditCases.value)
})

const assignedCases = computed<AuditCase[]>(() => {
  if (!auditorId.value) return []
  return assignedAuditCases(auditorId.value, auditCases.value)
})

const activeItem = computed<AuditItem | null>(() => {
  if (!activeAuditId.value) return null
  return assignedItems.value.find((i) => i.auditId === activeAuditId.value) ?? null
})


const activeCase = computed<AuditCase | null>(() => {
  if (!activeItem.value) return null
  if (activeItem.value.seq != null) {
    return auditCases.value.find((c) => c.seq === activeItem.value!.seq) ?? null
  }
  return auditCases.value.find((c) => c.caseId === activeItem.value!.caseId) ?? null
})

const activeSeq = computed(() => activeCase.value?.seq ?? activeItem.value?.seq ?? null)

const collaborationRaw = computed(() => {
  const log = activeCase.value?.collaborationLog as { raw?: string } | undefined
  return log?.raw
})

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
})

const collaborationHtml = computed(() => {
  if (!collaborationRaw.value) return null
  return md.render(collaborationRaw.value)
})

const activeFailureCode = computed(() => activeItem.value?.failureCode ?? activeItem.value?.taxonomyKey ?? null)
const activeFailureMode = computed(() => {
  const code = activeFailureCode.value
  if (!code) return null
  return FAILURE_MODE_DEFINITION_MAPPING[code] ?? null
})

watch(
  activeItem,
  (it) => {
    if (!it || !auditorId.value) return
    const existing = annotations.value[it.auditId]
    verdict.value = existing?.verdict ?? null
  },
  { immediate: true },
)

const doneCount = computed(() => Object.keys(annotations.value).length)
const isAllDone = computed(() => assignedItems.value.length > 0 && doneCount.value >= assignedItems.value.length)
const completionHintShown = ref(false)

/** Check if a case is fully done (all its items annotated). Uses seq as primary key. */
const isCaseDone = (c: AuditCase): boolean =>
  c.items.every((it) => Boolean(annotations.value[it.auditId]))

const isAllCasesDone = computed(
  () =>
    assignedCases.value.length > 0 &&
    assignedCases.value.every((c) => isCaseDone(c)),
)

const nextTodoAuditId = computed(() => {
  const list = assignedItems.value
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
  if (!verdict.value) return
  const annotation: AuditAnnotation = {
    auditId: activeItem.value.auditId,
    caseId: activeItem.value.caseId,
    seq: activeItem.value.seq,
    auditorId: auditorId.value,
    dataset: activeCase.value?.dataset,
    mas: activeCase.value?.framework,
    llm: activeCase.value?.llm,
    taxonomyKey: activeItem.value.taxonomyKey as any,
    verdict: verdict.value,
    updatedAt: new Date().toISOString(),
  }
  saveAudit(auditorId.value, annotation)
  annotations.value = loadAuditMap(auditorId.value)
}

watch(verdict, persistActive)

const next = () => {
  if (!verdict.value) {
    toast.value = { show: true, message: 'Please select a verdict (Yes/No) before moving on.' }
    window.setTimeout(() => {
      toast.value = { show: false, message: '' }
    }, 1500)
    return
  }
  activeAuditId.value = nextTodoAuditId.value
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

watch(isAllDone, (done) => {
  if (!done || completionHintShown.value) return
  completionHintShown.value = true
  toast.value = { show: true, message: 'All items reviewed. Please export your JSON.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 2000)
})

loadCases()

watch(
  auditCases,
  (list) => {
    if (!auditorId.value || list.length === 0) return
    const items = assignedAuditItems(auditorId.value, list)
    const currentValid = items.some((it) => it.auditId === activeAuditId.value)
    if ((!activeAuditId.value || !currentValid) && items.length > 0) {
      activeAuditId.value = items[0]!.auditId
    }
  },
  { deep: true },
)

watch(activeAuditId, () => {
  // Keep drawer state when switching items; close only popovers.
  isInstructionPopoverOpen.value = false
})

onMounted(() => {
  document.addEventListener('click', onDocumentClick)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', onDocumentClick)
})
const exportJson = () => {
  if (!auditorId.value) return
  const name = `Auditor_${auditorId.value}`

  const casesBySeq = new Map(auditCases.value.map((c) => [c.seq, c] as const).filter(([seq]) => seq != null))
  const casesById = new Map(auditCases.value.map((c) => [c.caseId, c]))
  const rawAnnotations = loadAuditMap(auditorId.value)
  const enrichedAnnotations = Object.fromEntries(
    Object.entries(rawAnnotations).map(([auditId, ann]) => {
      const c = (ann.seq != null ? casesBySeq.get(ann.seq) : null) ?? casesById.get(ann.caseId)
      return [
        auditId,
        {
          ...ann,
          dataset: ann.dataset ?? c?.dataset,
          mas: ann.mas ?? c?.framework,
          llm: ann.llm ?? c?.llm,
        },
      ]
    }),
  )

  const payload = {
    schema: 'medagentaudit.audit.v1',
    auditor: { id: auditorId.value, name },
    exportedAt: new Date().toISOString(),
    annotations: enrichedAnnotations,
  }
  downloadJson(`${name}_audit.json`, payload)
}

const copyLog = async () => {
  const raw = collaborationRaw.value
  if (!raw) return
  await copyToClipboard(raw)
  toast.value = { show: true, message: 'Copied Collaboration Text.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 1200)
}

const copyInstruction = async () => {
  const text = activeItem.value?.instructionText ?? activeItem.value?.context
  if (!text) return
  await copyToClipboard(text)
  toast.value = { show: true, message: 'Copied Instruction Text.' }
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

const copyFailureMode = async () => {
  const code = activeFailureCode.value
  const mode = activeFailureMode.value
  if (!code && !mode) return
  const title = `${code ?? ''}${code && mode?.name ? ': ' : ''}${mode?.name ?? ''}`.trim()
  const definition = mode?.definition ?? ''
  const payload = `${title}\n\n${definition}`.trim()
  if (!payload) return
  await copyToClipboard(payload)
  toast.value = { show: true, message: 'Copied Failure Mode.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 1200)
}

const toggleInstructionPopover = () => {
  isInstructionPopoverOpen.value = !isInstructionPopoverOpen.value
}

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
            <div class="text-sm font-semibold text-slate-900">Auditor</div>
            <div class="mt-2">
              <AppSelect
                v-model="auditorIdText"
                :options="auditorOptions"
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
                Next TODO
              </AppButton>
              <div class="flex items-center gap-2">
                <AppButton variant="secondary" :disabled="!auditorId" @click="exportJson">Export JSON</AppButton>
                <span v-if="isAllCasesDone" class="text-xs font-medium text-emerald-700">All done</span>
              </div>
            </div>
          </div>
        </div>

        <div class="mt-4 max-h-[60vh] min-h-[200px] overflow-auto pr-1">
          <div v-if="!auditorId" class="rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-600">
            Select an auditor ID to load assigned items.
          </div>

          <div v-else class="space-y-2">
            <button
              v-for="c in assignedCases"
              :key="c.seq ?? c.caseId"
              type="button"
              class="w-full rounded-xl border p-3 text-left text-sm transition"
              :class="activeSeq === c.seq ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-white hover:bg-slate-50'"
              @click="activeAuditId = c.items[0]?.auditId ?? assignedItems.find((it) => it.seq === c.seq)?.auditId ?? null"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="truncate font-medium text-slate-900">Case {{ c.seq }}: {{ c.caseId }}</div>
                <div
                  class="shrink-0 rounded-md px-2 py-0.5 text-xs"
                  :class="isCaseDone(c) ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-600'"
                >
                  {{ isCaseDone(c) ? 'Done' : 'TODO' }}
                </div>
              </div>
              <div class="mt-1 text-xs text-slate-600">{{ c.dataset }} · {{ c.framework }} · {{ c.modality }}</div>
            </button>
          </div>
        </div>
      </AppCard>
    </div>

    <div class="min-w-0">
      <div v-if="activeItem && activeCase" class="grid gap-4 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)_minmax(0,420px)]">
        <div>
          <AppCard class="max-h-[86vh] overflow-auto p-5">
            <div class="flex flex-wrap items-start justify-between gap-2">
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
        </div>

        <div>
	          <AppCard class="max-h-[86vh] overflow-visible p-5">
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
	                    <div class="whitespace-pre-wrap">{{ activeItem.instructionText ?? activeItem.context }}</div>
	                  </div>
	                </div>
	                <AppButton variant="secondary" @click="copyLog">Copy Collaboration</AppButton>
	              </div>
	            </div>

	            <div
	              v-if="collaborationHtml"
	              class="mt-3 max-h-[75vh] overflow-auto rounded-xl border border-slate-200 bg-white p-4"
	            >
	              <div class="prose prose-slate max-w-none text-sm text-slate-900" v-html="collaborationHtml"></div>
	            </div>

            <div v-else class="mt-3 text-xs text-slate-600">No collaboration log found.</div>
          </AppCard>
        </div>

        <div>
          <AppCard class="max-h-[86vh] overflow-auto p-5">
        <div class="space-y-4">
          <div>
            <div class="flex items-center justify-between gap-3">
              <div class="text-sm font-semibold text-slate-900">Failure mode</div>
              <AppButton variant="secondary" @click="copyFailureMode">Copy Failure Mode</AppButton>
            </div>
            <div class="mt-2 rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-800">
              <div class="flex flex-wrap items-center gap-2">
                <span class="font-medium text-slate-900">{{ activeFailureCode }}: {{ activeFailureMode?.name ?? '' }}</span>
              </div>
              <div class="mt-2 text-sm text-slate-600">{{ activeFailureMode?.definition ?? '' }}</div>
            </div>
          </div>

          <div class="flex items-center justify-between">
            <div class="text-sm font-semibold text-slate-900">Verdict</div>
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

          <AppButton variant="secondary" class="w-full" @click="next">Next TODO</AppButton>
        </div>
          </AppCard>
        </div>
      </div>

      <AppCard v-else class="p-5">
        <div v-if="!casesLoaded" class="text-sm text-slate-600">Loading audit cases...</div>
        <div v-else class="text-sm text-slate-600">No assigned audit items. Select an auditor and item.</div>
      </AppCard>
    </div>
  </div>

  <AppToast :show="toast.show" :message="toast.message" />
</template>
