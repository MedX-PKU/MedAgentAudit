<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { TAXONOMY, type TaxonomyKey } from '../../../domain/taxonomy'
import AppButton from '../../../components/ui/AppButton.vue'
import AppCard from '../../../components/ui/AppCard.vue'
import AppSelect from '../../../components/ui/AppSelect.vue'
import ProgressBar from '../../../components/annotation/ProgressBar.vue'
import AppToast from '../../../components/ui/AppToast.vue'
import type { AuditAnnotation, AuditCase, AuditItem } from '../../../domain/types'
import { downloadJson } from '../../../lib/download'
import { copyToClipboard } from '../../../lib/clipboard'
import { loadAuditCases } from '../../../data/audit/cases'
import { assignedAuditCases, assignedAuditItems, type AuditorId } from './auditAssignment'
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
const verdict = ref<Verdict | null>(null)
const toast = ref<{ show: boolean; message: string }>({ show: false, message: '' })

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
    if (!activeAuditId.value && items.length > 0) activeAuditId.value = items[0]!.auditId
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

const activeCaseId = computed(() => activeCase.value?.caseId ?? null)

const activeCase = computed<AuditCase | null>(() => {
  if (!activeItem.value) return null
  return auditCases.value.find((c) => c.caseId === activeItem.value!.caseId) ?? null
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

const collaborationRaw = computed(() => {
  const log = activeCase.value?.collaborationLog as { raw?: string } | undefined
  return log?.raw
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
    verdict.value = existing?.verdict ?? null
  },
  { immediate: true },
)

const doneCount = computed(() => Object.keys(annotations.value).length)
const isAllDone = computed(() => assignedItems.value.length > 0 && doneCount.value >= assignedItems.value.length)
const completionHintShown = ref(false)

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
    taxonomyKey: activeItem.value.taxonomyKey as TaxonomyKey,
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
    if (!activeAuditId.value && auditorId.value && list.length > 0) {
      const items = assignedAuditItems(auditorId.value, list)
      activeAuditId.value = items[0]?.auditId ?? null
    }
  },
  { deep: true },
)
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

const copyLog = async () => {
  const raw = collaborationRaw.value
  if (!raw) return
  await copyToClipboard(raw)
  toast.value = { show: true, message: 'Copied collaboration log.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 1200)
}

const copyQuestion = async () => {
  if (!activeCase.value) return
  const questionText = activeCase.value.question ?? ''
  const optionsText = (activeCase.value.options ?? []).join('\n')
  const payload = `Question：${questionText}\nOptions：${optionsText}`
  await copyToClipboard(payload)
  toast.value = { show: true, message: 'Copied question + options.' }
  window.setTimeout(() => {
    toast.value = { show: false, message: '' }
  }, 1200)
}

const copyImage = async () => {
  const url = activeCase.value?.image?.path
  if (!url) return
  try {
    toast.value = { show: true, message: 'Copying image...' }
    const res = await fetch(url)
    const blob = await res.blob()
    if (typeof ClipboardItem === 'undefined' || !navigator.clipboard?.write) {
      await copyToClipboard(url)
      toast.value = { show: true, message: 'Copied image URL.' }
      window.setTimeout(() => {
        toast.value = { show: false, message: '' }
      }, 1200)
      return
    }
    await navigator.clipboard.write([new ClipboardItem({ [blob.type || 'image/png']: blob })])
    toast.value = { show: true, message: 'Copied image.' }
    window.setTimeout(() => {
      toast.value = { show: false, message: '' }
    }, 1200)
  } catch (error) {
    console.error(error)
    await copyToClipboard(url)
    toast.value = { show: true, message: 'Copied image URL.' }
    window.setTimeout(() => {
      toast.value = { show: false, message: '' }
    }, 1200)
  }
}
</script>

<template>
  <div class="space-y-4">
    <div class="group fixed left-0 top-0 z-40 flex h-full items-center">
      <div class="pointer-events-none h-full w-[2px] bg-slate-900/5 transition group-hover:bg-slate-900/10"></div>
      <div
        class="pointer-events-auto ml-2 w-[320px] -translate-x-full rounded-2xl border border-slate-200 bg-white p-4 shadow-xl transition group-hover:translate-x-0"
      >
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
              v-for="c in assignedCases"
              :key="c.caseId"
              type="button"
              class="w-full rounded-xl border p-3 text-left text-sm transition"
              :class="c.caseId === activeCaseId ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-white hover:bg-slate-50'"
              @click="activeAuditId = assignedItems.find((it) => it.caseId === c.caseId)?.auditId ?? null"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="truncate font-medium text-slate-900">{{ c.caseId }}</div>
              </div>
              <div class="mt-1 text-xs text-slate-600">{{ c.dataset }} · {{ c.framework }} · {{ c.modality }}</div>
            </button>
          </div>
        </div>
      </AppCard>
      </div>
    </div>

    <div class="min-w-0">
      <div v-if="activeItem && activeCase" class="grid gap-4 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)_minmax(0,420px)]">
        <div>
          <AppCard class="max-h-[86vh] overflow-auto p-5">
            <div class="flex flex-wrap items-start justify-between gap-2">
              <div>
                <div class="mt-1 text-lg font-semibold text-slate-900">Audit {{ activeItem.auditId }}</div>
              </div>
            </div>

	            <div class="mt-4 space-y-4">
	              <div>
	                <div class="flex items-center justify-between gap-3">
	                  <div class="text-sm font-semibold text-slate-900">Question</div>
	                  <AppButton variant="secondary" @click="copyQuestion">Copy</AppButton>
	                </div>
	                <div class="mt-2 whitespace-pre-wrap rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-800">
	                  {{ activeCase.question }}
	                </div>
	              </div>

              <div v-if="activeCase.modality === 'vqa' && activeCase.image?.path">
                <div class="flex items-center justify-between gap-3">
                  <div class="text-sm font-semibold text-slate-900">Image</div>
                  <AppButton variant="secondary" @click="copyImage">Copy</AppButton>
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
	          <AppCard class="max-h-[86vh] overflow-auto p-5">
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

            <div class="mt-4">
              <div class="flex items-center justify-between gap-3">
                <div class="text-sm font-semibold text-slate-900">Instruction Text</div>
                <div class="group relative">
                  <AppButton variant="secondary">Instruction Text</AppButton>
	                  <div
	                    class="pointer-events-none absolute right-0 top-full z-50 mt-2 hidden w-[520px] whitespace-pre-wrap rounded-xl border border-slate-200 bg-white p-3 text-xs text-slate-900 shadow-lg group-hover:block"
	                  >
	                    {{ activeItem.instructionText ?? activeItem.context }}
	                  </div>
                </div>
              </div>
            </div>

	            <div class="mt-4 flex items-center justify-between gap-3">
	              <div class="text-sm font-semibold text-slate-900">Collaboration log</div>
	              <AppButton variant="secondary" @click="copyLog">Copy</AppButton>
	            </div>

	            <div v-if="collaborationRaw" class="mt-3 whitespace-pre-wrap rounded-lg bg-white p-3 text-xs text-slate-900">
	              {{ collaborationRaw }}
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
                  <div class="text-xs text-slate-700">Answer: {{ round.synthesis.parsed_output.answer ?? '-' }}</div>
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
                  <div class="text-xs text-slate-700">Answer: {{ round.decision.parsed_output.answer ?? '-' }}</div>
                  <div v-if="round.decision.parsed_output.explanation" class="text-xs text-slate-600 whitespace-pre-wrap">
                    {{ round.decision.parsed_output.explanation }}
                  </div>
                </div>
              </AppCard>
            </div>

            <div v-else class="mt-3 text-xs text-slate-600">No structured collaboration log found.</div>
          </AppCard>
        </div>

        <div>
          <AppCard class="max-h-[86vh] overflow-auto p-5">
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

          <AppButton variant="secondary" class="w-full" @click="next">Next</AppButton>
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
