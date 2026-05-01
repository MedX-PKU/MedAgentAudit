<script setup lang="ts">
import { computed, ref } from 'vue'

import AppCard from '../../components/ui/AppCard.vue'
import AppShell from '../../components/layout/AppShell.vue'
import {
  annotationCards,
  benchmarkMatrix,
  phaseDeck,
  projectFacts,
  projectMeta,
  resultDeck,
  sectionNav,
  studyStages,
} from '../../content/site'

const activeStudyId = ref(studyStages[0]?.id ?? '')
const activePhaseId = ref(phaseDeck[0]?.id ?? '')
const activeResultId = ref(resultDeck[0]?.id ?? '')
const firstStudy = studyStages[0]!
const firstPhase = phaseDeck[0]!
const firstResult = resultDeck[0]!

const switcherButtonClass = (isActive: boolean) =>
  isActive
    ? 'border-sky-300 bg-sky-50 text-sky-900 shadow-[0_14px_32px_rgba(56,189,248,0.14)] ring-4 ring-sky-100'
    : 'border-slate-200 bg-white/86 text-slate-600 hover:-translate-y-0.5 hover:border-slate-300 hover:bg-white hover:text-slate-950'

const activeStudy = computed(() => studyStages.find((stage) => stage.id === activeStudyId.value) ?? firstStudy)
const activeStudyIndex = computed(() => studyStages.findIndex((stage) => stage.id === activeStudy.value.id) + 1)
const activePhase = computed(() => phaseDeck.find((phase) => phase.id === activePhaseId.value) ?? firstPhase)
const activeResult = computed(() => resultDeck.find((item) => item.id === activeResultId.value) ?? firstResult)

const isNumericAuthorMark = (mark: string) => /^\d+$/.test(mark)
const formatAuthorMarks = (marks: string[]) => {
  const numericMarks = marks.filter(isNumericAuthorMark)
  const symbolicMarks = marks.filter((mark) => !isNumericAuthorMark(mark))
  return [...numericMarks, ...symbolicMarks]
}

const formattedAuthors = computed(() =>
  projectMeta.authors.map((author) => ({
    ...author,
    formattedMarks: formatAuthorMarks(author.marks),
  })),
)

const linkClass = (kind: 'primary' | 'secondary') =>
  kind === 'primary'
    ? 'bg-slate-950 text-white shadow-[0_16px_34px_rgba(15,23,42,0.16)] hover:bg-slate-800'
    : 'border border-slate-200 bg-white/88 text-slate-900 hover:border-slate-300 hover:bg-white'
</script>

<template>
  <AppShell content-width="wide">
    <section id="overview" class="py-7 sm:py-8 xl:py-10">
      <div class="mx-auto max-w-[1280px] space-y-7">
        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2 md:mx-auto xl:mx-0">
            <a
              v-for="item in sectionNav"
              :key="item.id"
              :href="`#${item.id}`"
              class="rounded-full border border-slate-200 bg-white/84 px-4 py-2 text-sm font-semibold text-slate-600 transition hover:border-slate-300 hover:text-slate-950"
            >
              {{ item.label }}
            </a>
          </div>
        </div>

        <div class="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
          <div class="space-y-5">
            <div class="inline-flex rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-semibold tracking-[0.14em] text-teal-800 uppercase">
              Medical MAS collaboration audit
            </div>

            <h1 class="max-w-[28ch] text-4xl font-semibold leading-[1.02] text-slate-950 sm:text-5xl xl:text-6xl">
              {{ projectMeta.title }}
            </h1>

            <div class="max-w-[72rem] space-y-4">
              <div class="text-[0.98rem] leading-[1.82] text-slate-900 sm:text-[1.02rem] lg:text-[1.06rem]">
                <template v-for="(author, index) in formattedAuthors" :key="author.name">
                  <span class="inline-flex items-start">
                    <span>{{ author.name }}</span>
                    <span
                      v-if="author.formattedMarks.length"
                      class="ml-1 inline-flex -translate-y-[0.38em] items-start gap-[0.14rem] text-[0.66em] font-semibold tracking-[0.02em] text-slate-500"
                    >
                      <template v-for="(mark, markIndex) in author.formattedMarks" :key="`${author.name}-${mark}-${markIndex}`">
                        <span :class="isNumericAuthorMark(mark) ? 'text-slate-500' : 'text-slate-900'">{{ mark }}</span>
                        <span v-if="markIndex < author.formattedMarks.length - 1" class="text-slate-400">,</span>
                      </template>
                    </span>
                  </span>
                  <span v-if="index < formattedAuthors.length - 1">, </span>
                </template>
              </div>

              <div class="max-w-[72rem] space-y-2.5 border-t border-slate-200/80 pt-3 text-sm leading-6 text-slate-600">
                <div v-for="item in projectMeta.affiliations" :key="item.mark" class="flex gap-2.5">
                  <span class="w-4 shrink-0 font-semibold text-slate-950">{{ item.mark }}</span>
                  <span>{{ item.text }}</span>
                </div>

                <div class="space-y-1.5 pt-1">
                  <div v-for="item in projectMeta.notes" :key="item.mark" class="flex gap-2.5">
                    <span class="w-4 shrink-0 font-semibold text-slate-950">{{ item.mark }}</span>
                    <span>{{ item.text }}</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="flex flex-wrap gap-3">
              <template v-for="link in projectMeta.links" :key="link.href">
                <RouterLink
                  v-if="link.href.startsWith('/')"
                  :to="link.href"
                  class="inline-flex items-center justify-center rounded-full px-4 py-2 text-sm font-semibold transition"
                  :class="linkClass(link.kind)"
                >
                  {{ link.label }}
                </RouterLink>
                <a
                  v-else
                  :href="link.href"
                  target="_blank"
                  rel="noreferrer"
                  class="inline-flex items-center justify-center rounded-full px-4 py-2 text-sm font-semibold transition"
                  :class="linkClass(link.kind)"
                >
                  {{ link.label }}
                </a>
              </template>
            </div>
          </div>

          <AppCard class="border-slate-200/80 bg-white/86 !p-5">
            <div class="space-y-5">
              <div>
                <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Audit frame</div>
                <div class="mt-2 text-2xl font-semibold leading-tight text-slate-950">Trace-level probes across collaboration.</div>
              </div>

              <div class="space-y-3">
                <div
                  v-for="(phase, index) in phaseDeck"
                  :key="phase.id"
                  class="rounded-lg border border-slate-200 bg-slate-50/80 p-4"
                >
                  <div class="flex items-center gap-3">
                    <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-950 text-sm font-semibold text-white">
                      {{ index + 1 }}
                    </div>
                    <div>
                      <div class="text-sm font-semibold text-slate-950">{{ phase.kicker }}</div>
                      <div class="text-xs leading-5 text-slate-500">{{ phase.label }}</div>
                    </div>
                  </div>
                  <p class="mt-3 text-sm leading-7 text-slate-600">{{ phase.title }}</p>
                </div>
              </div>
            </div>
          </AppCard>
        </div>

        <div class="mx-auto max-w-[72rem] space-y-4 pt-2">
          <div class="text-center font-display text-2xl font-semibold text-slate-950">
            Abstract
          </div>
          <AppCard class="border-slate-200/80 bg-white/84 !p-5 text-left sm:!p-6">
            <p class="text-base leading-8 text-pretty text-slate-600 sm:text-lg sm:leading-9">
              {{ projectMeta.abstract }}
            </p>
          </AppCard>
        </div>

        <div class="grid gap-3 lg:grid-cols-3">
          <AppCard v-for="fact in projectFacts" :key="fact.label" class="border-slate-200/80 bg-white/86 !p-5">
            <div class="space-y-3">
              <div class="text-[11px] font-semibold tracking-[0.18em] text-slate-500 uppercase">{{ fact.label }}</div>
              <div class="text-2xl font-semibold text-slate-950">{{ fact.value }}</div>
              <p class="text-sm leading-7 text-slate-600">{{ fact.note }}</p>
            </div>
          </AppCard>
        </div>
      </div>
    </section>

    <section id="design" class="py-8 sm:py-10">
      <div class="mx-auto max-w-[1280px] space-y-5">
        <div class="mx-auto max-w-[58rem] space-y-3 text-center">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Study Design</div>
          <h2 class="font-display text-3xl font-semibold leading-tight text-slate-950 sm:text-4xl xl:text-5xl">
            Baseline comparison, taxonomy development, and large-scale auditing.
          </h2>
          <p class="text-base leading-8 text-slate-600 sm:text-lg">
            The study connects final-answer performance to audited interaction trajectories so failures can be localized by phase and interaction step.
          </p>
        </div>

        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2 md:mx-auto" role="tablist" aria-label="Study design stages">
            <button
              v-for="stage in studyStages"
              :key="stage.id"
              :id="`study-tab-${stage.id}`"
              :aria-selected="activeStudyId === stage.id"
              type="button"
              role="tab"
              class="rounded-full border px-4 py-2 text-sm font-semibold transition"
              :class="switcherButtonClass(activeStudyId === stage.id)"
              @click="activeStudyId = stage.id"
            >
              {{ stage.shortLabel }}
            </button>
          </div>
        </div>

        <AppCard class="border-slate-200/80 bg-white/86" role="tabpanel" :aria-labelledby="`study-tab-${activeStudy.id}`">
          <div class="space-y-6 p-5 sm:p-6">
            <div class="flex flex-wrap items-center justify-center gap-3 text-sm text-slate-500">
              <span class="rounded-full bg-sky-600 px-3 py-1 text-xs font-semibold tracking-[0.16em] text-white uppercase">
                Step {{ activeStudyIndex }}
              </span>
              <span class="rounded-full border border-slate-200 bg-white px-3 py-1 font-semibold text-slate-700">
                {{ activeStudy.kicker }}
              </span>
            </div>

            <div class="mx-auto max-w-[60rem] space-y-3 text-center">
              <h3 class="font-display text-3xl font-semibold leading-tight text-slate-950 sm:text-4xl">
                {{ activeStudy.title }}
              </h3>
              <p class="text-base leading-8 text-slate-600 sm:text-lg">
                {{ activeStudy.description }}
              </p>
            </div>

            <div class="grid gap-3 sm:grid-cols-3">
              <div v-for="metric in activeStudy.metrics" :key="`${activeStudy.id}-${metric.label}`" class="rounded-lg border border-slate-200 bg-slate-50/90 px-4 py-4">
                <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">{{ metric.label }}</div>
                <div class="mt-3 text-2xl font-semibold text-slate-950">{{ metric.value }}</div>
              </div>
            </div>
          </div>
        </AppCard>

        <div class="grid gap-3 lg:grid-cols-4">
          <div v-for="item in benchmarkMatrix" :key="item.label" class="rounded-lg border border-slate-200 bg-white/82 p-4">
            <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">{{ item.label }}</div>
            <div class="mt-3 text-sm leading-7 text-slate-700">{{ item.value }}</div>
          </div>
        </div>
      </div>
    </section>

    <section id="taxonomy" class="py-8 sm:py-10">
      <div class="mx-auto max-w-[1280px] space-y-5">
        <div class="mx-auto max-w-[58rem] space-y-3 text-center">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Taxonomy</div>
          <h2 class="font-display text-3xl font-semibold leading-tight text-slate-950 sm:text-4xl xl:text-5xl">
            Ten collaborative failure modes across three phases.
          </h2>
          <p class="text-base leading-8 text-slate-600 sm:text-lg">
            Each phase defines what context the auditor should read and which interaction failure is being probed.
          </p>
        </div>

        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2 md:mx-auto" role="tablist" aria-label="Taxonomy phases">
            <button
              v-for="phase in phaseDeck"
              :key="phase.id"
              :id="`phase-tab-${phase.id}`"
              :aria-selected="activePhaseId === phase.id"
              type="button"
              role="tab"
              class="rounded-full border px-4 py-2 text-sm font-semibold transition"
              :class="switcherButtonClass(activePhaseId === phase.id)"
              @click="activePhaseId = phase.id"
            >
              {{ phase.label }}
            </button>
          </div>
        </div>

        <AppCard class="border-slate-200/80 bg-white/86" role="tabpanel" :aria-labelledby="`phase-tab-${activePhase.id}`">
          <div class="grid gap-5 p-5 sm:p-6 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
            <div class="space-y-3">
              <div class="inline-flex rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-[11px] font-semibold tracking-[0.16em] text-teal-800 uppercase">
                {{ activePhase.kicker }}
              </div>
              <h3 class="font-display text-3xl font-semibold leading-tight text-slate-950 sm:text-4xl">
                {{ activePhase.title }}
              </h3>
              <p class="text-base leading-8 text-slate-600">{{ activePhase.summary }}</p>
            </div>

            <div class="grid gap-3 sm:grid-cols-2">
              <div v-for="mode in activePhase.modes" :key="mode.code" class="rounded-lg border border-slate-200 bg-slate-50/90 p-4">
                <div class="flex items-center justify-between gap-3">
                  <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">{{ mode.code }}</div>
                  <div class="rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-slate-900 ring-1 ring-slate-200">
                    {{ mode.rate }}
                  </div>
                </div>
                <div class="mt-3 text-base font-semibold text-slate-950">{{ mode.label }}</div>
              </div>
            </div>
          </div>
        </AppCard>
      </div>
    </section>

    <section id="results" class="py-8 sm:py-10">
      <div class="mx-auto max-w-[1280px] space-y-5">
        <div class="mx-auto max-w-[58rem] space-y-3 text-center">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Results</div>
          <h2 class="font-display text-3xl font-semibold leading-tight text-slate-950 sm:text-4xl xl:text-5xl">
            Failure trajectories explain where collaboration breaks down.
          </h2>
          <p class="text-base leading-8 text-slate-600 sm:text-lg">
            The public summary focuses on failure entry, persistence during discussion, decision-making bias, and auditor validation.
          </p>
        </div>

        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2 md:mx-auto" role="tablist" aria-label="Result sections">
            <button
              v-for="item in resultDeck"
              :key="item.id"
              :id="`result-tab-${item.id}`"
              :aria-selected="activeResultId === item.id"
              type="button"
              role="tab"
              class="rounded-full border px-4 py-2 text-sm font-semibold transition"
              :class="switcherButtonClass(activeResultId === item.id)"
              @click="activeResultId = item.id"
            >
              {{ item.label }}
            </button>
          </div>
        </div>

        <AppCard class="border-slate-200/80 bg-white/86" role="tabpanel" :aria-labelledby="`result-tab-${activeResult.id}`">
          <div class="space-y-6 p-5 sm:p-6">
            <div class="mx-auto max-w-[60rem] space-y-3 text-center">
              <h3 class="font-display text-3xl font-semibold leading-tight text-slate-950 sm:text-4xl">
                {{ activeResult.title }}
              </h3>
              <p class="text-base leading-8 text-slate-600 sm:text-lg">
                {{ activeResult.summary }}
              </p>
            </div>

            <div class="grid gap-3 sm:grid-cols-3">
              <div v-for="stat in activeResult.stats" :key="`${activeResult.id}-${stat.label}`" class="rounded-lg border border-slate-200 bg-slate-50/90 px-4 py-4">
                <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">{{ stat.label }}</div>
                <div class="mt-3 text-2xl font-semibold text-slate-950">{{ stat.value }}</div>
              </div>
            </div>
          </div>
        </AppCard>
      </div>
    </section>

    <section id="annotation" class="py-8 sm:py-10">
      <div class="mx-auto max-w-[1280px] space-y-5">
        <div class="mx-auto max-w-[58rem] space-y-3 text-center">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Annotation</div>
          <h2 class="font-display text-3xl font-semibold leading-tight text-slate-950 sm:text-4xl xl:text-5xl">
            Browser-only tools for the human evaluation workflow.
          </h2>
          <p class="text-base leading-8 text-slate-600 sm:text-lg">
            The annotation interface keeps the current open-coding and audit tasks while using local browser storage and JSON export.
          </p>
        </div>

        <div class="grid gap-3 md:grid-cols-2">
          <RouterLink
            v-for="card in annotationCards"
            :key="card.route"
            :to="card.route"
            class="group rounded-lg border border-slate-200/80 bg-white/86 p-5 shadow-[0_18px_50px_rgba(15,23,42,0.06)] transition hover:-translate-y-0.5 hover:border-sky-200 hover:bg-white"
          >
            <div class="space-y-3">
              <div class="text-[11px] font-semibold tracking-[0.18em] text-sky-700 uppercase">{{ card.meta }}</div>
              <div class="text-2xl font-semibold text-slate-950">{{ card.title }}</div>
              <p class="text-sm leading-7 text-slate-600">{{ card.body }}</p>
              <div class="text-sm font-semibold text-slate-950 group-hover:text-sky-900">Open {{ card.title }}</div>
            </div>
          </RouterLink>
        </div>
      </div>
    </section>
  </AppShell>
</template>
