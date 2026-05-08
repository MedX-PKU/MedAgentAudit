<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import AppCard from '../../components/ui/AppCard.vue'
import AppShell from '../../components/layout/AppShell.vue'
import {
  annotationCards,
  benchmarkMatrix,
  keyFindings,
  phaseDeck,
  projectFacts,
  projectMeta,
  sectionNav,
  studyStages,
} from '../../content/site'

const activeStudyId = ref(studyStages[0]?.id ?? '')
const activePhaseId = ref(phaseDeck[0]?.id ?? '')
const activeSectionId = ref(sectionNav[0]?.id ?? 'overview')
const firstStudy = studyStages[0]!
const firstPhase = phaseDeck[0]!
let sectionObserver: IntersectionObserver | null = null

const switcherButtonClass = (isActive: boolean) =>
  isActive
    ? 'border-sky-200 bg-sky-50/80 text-sky-900 ring-2 ring-sky-100/80'
    : 'border-slate-200 bg-white/78 text-slate-600 hover:border-slate-300 hover:bg-white hover:text-slate-950'

const activeStudy = computed(() => studyStages.find((stage) => stage.id === activeStudyId.value) ?? firstStudy)
const activeStudyIndex = computed(() => studyStages.findIndex((stage) => stage.id === activeStudy.value.id) + 1)
const activePhase = computed(() => phaseDeck.find((phase) => phase.id === activePhaseId.value) ?? firstPhase)

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
    ? 'border border-slate-400 bg-white text-slate-950 hover:border-slate-950 hover:bg-slate-50'
    : 'border border-slate-200 bg-white/82 text-slate-700 hover:border-slate-400 hover:bg-white hover:text-slate-950'

const sectionLinkClass = (isActive: boolean) =>
  isActive
    ? 'border-slate-950 text-slate-950'
    : 'border-transparent text-slate-500 hover:border-slate-950 hover:text-slate-950'

onMounted(() => {
  sectionObserver = new IntersectionObserver(
    (entries) => {
      const visibleEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((first, second) => second.intersectionRatio - first.intersectionRatio)[0]

      if (visibleEntry?.target.id) {
        activeSectionId.value = visibleEntry.target.id
      }
    },
    {
      rootMargin: '-28% 0px -58% 0px',
      threshold: [0, 0.2, 0.5],
    },
  )

  sectionNav.forEach((item) => {
    const section = document.getElementById(item.id)
    if (section) sectionObserver?.observe(section)
  })
})

onBeforeUnmount(() => {
  sectionObserver?.disconnect()
})
</script>

<template>
  <AppShell content-width="wide">
    <div class="mx-auto grid max-w-[1460px] gap-8 lg:grid-cols-[150px_minmax(0,1fr)] xl:grid-cols-[170px_minmax(0,1fr)]">
      <aside class="hidden lg:block">
        <nav class="sticky top-28 space-y-1 border-l border-slate-200 pl-4" aria-label="Paper sections">
          <a
            v-for="item in sectionNav"
            :key="item.id"
            :href="`#${item.id}`"
            class="block border-l-2 px-3 py-2 text-sm font-medium transition"
            :class="sectionLinkClass(activeSectionId === item.id)"
          >
            {{ item.label }}
          </a>
        </nav>
      </aside>

      <div class="min-w-0">
        <div class="sticky top-[4.8rem] z-40 -mx-2.5 border-y border-slate-200/80 bg-white/90 px-2.5 backdrop-blur-xl sm:-mx-3 sm:px-3 lg:hidden">
          <nav class="flex gap-1 overflow-x-auto py-2" aria-label="Paper sections">
            <a
              v-for="item in sectionNav"
              :key="item.id"
              :href="`#${item.id}`"
              class="shrink-0 border-b-2 px-3 py-2 text-sm font-medium transition"
              :class="sectionLinkClass(activeSectionId === item.id)"
            >
              {{ item.label }}
            </a>
          </nav>
        </div>

        <section id="overview" class="scroll-mt-36 py-8 sm:py-10 xl:py-12">
      <div class="mx-auto max-w-[1180px] space-y-7 sm:space-y-8">
        <div class="space-y-5">
          <h1 class="mx-auto max-w-[1180px] text-center text-[2rem] font-medium leading-[1.13] text-slate-950 sm:text-[2.35rem] lg:text-[2.2rem] xl:text-[2.42rem] 2xl:text-[2.62rem]">
            <span v-for="line in projectMeta.titleLines" :key="line" class="block xl:whitespace-nowrap">
              {{ line }}
            </span>
          </h1>

          <div class="mx-auto max-w-[960px] space-y-3 text-slate-900">
            <div class="flex flex-wrap items-baseline justify-center gap-x-2 gap-y-1 text-[0.98rem] leading-7 sm:text-base">
              <template v-for="(author, index) in formattedAuthors" :key="author.name">
                <span class="inline-flex items-baseline">
                  <span class="font-medium text-slate-950">{{ author.name }}</span>
                  <sup v-if="author.formattedMarks.length" class="ml-0.5 text-[0.68rem] font-medium leading-none text-slate-500">
                    <template v-for="(mark, markIndex) in author.formattedMarks" :key="`${author.name}-${mark}-${markIndex}`">
                      <span :class="isNumericAuthorMark(mark) ? 'text-slate-500' : 'text-slate-800'">{{ mark }}</span>
                      <span v-if="markIndex < author.formattedMarks.length - 1">,</span>
                    </template>
                  </sup>
                </span>
                <span v-if="index < formattedAuthors.length - 1" class="text-slate-400">/</span>
              </template>
            </div>

            <div class="space-y-1.5 border-y border-slate-200 py-3 text-center text-sm leading-6 text-slate-700">
              <div v-for="item in projectMeta.affiliations" :key="item.mark" class="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-2">
                <sup class="pt-1 text-[0.72rem] font-medium leading-none text-slate-950">{{ item.mark }}</sup>
                <span class="text-left">{{ item.text }}</span>
              </div>
            </div>

            <div class="flex flex-wrap justify-center gap-x-5 gap-y-1 text-sm leading-6 text-slate-600">
              <div v-for="item in projectMeta.notes" :key="item.mark" class="inline-flex gap-2">
                <span class="font-medium text-slate-950">{{ item.mark }}</span>
                <span>{{ item.text }}</span>
              </div>
            </div>
          </div>

          <div class="flex flex-wrap justify-center gap-3">
            <template v-for="link in projectMeta.links" :key="link.href">
              <RouterLink
                v-if="link.href.startsWith('/')"
                :to="link.href"
                class="inline-flex items-center justify-center rounded-full px-4 py-2 text-sm font-medium transition"
                :class="linkClass(link.kind)"
              >
                {{ link.label }}
              </RouterLink>
              <a
                v-else
                :href="link.href"
                target="_blank"
                rel="noreferrer"
                class="inline-flex items-center justify-center rounded-full px-4 py-2 text-sm font-medium transition"
                :class="linkClass(link.kind)"
              >
                {{ link.label }}
              </a>
            </template>
          </div>
        </div>

        <div class="grid border-y border-slate-200 bg-white/54 sm:grid-cols-3">
          <div v-for="fact in projectFacts" :key="fact.label" class="border-slate-200 px-5 py-4 sm:border-l first:sm:border-l-0">
            <div class="text-sm text-slate-500">{{ fact.label }}</div>
            <div class="mt-1 text-[1.65rem] font-medium leading-tight text-slate-950">{{ fact.value }}</div>
            <p class="mt-2 text-sm leading-6 text-slate-600">{{ fact.note }}</p>
          </div>
        </div>
      </div>
    </section>

    <section id="design" class="scroll-mt-36 py-8 sm:py-10">
      <div class="mx-auto max-w-[1180px] space-y-5">
        <div class="mx-auto max-w-[54rem] space-y-2 text-center">
          <div class="text-sm font-medium text-slate-600">Study Design</div>
          <h2 class="font-display text-2xl font-medium leading-tight text-slate-950 sm:text-3xl xl:text-[2.2rem]">
            Baseline comparison, taxonomy development, and large-scale auditing.
          </h2>
          <p class="text-base leading-7 text-slate-600">
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
              class="rounded-full border px-4 py-2 text-sm font-medium transition"
              :class="switcherButtonClass(activeStudyId === stage.id)"
              @click="activeStudyId = stage.id"
            >
              {{ stage.shortLabel }}
            </button>
          </div>
        </div>

        <AppCard class="border-slate-200/80 bg-white/80" role="tabpanel" :aria-labelledby="`study-tab-${activeStudy.id}`">
          <div class="space-y-6 p-5 sm:p-6">
            <div class="flex flex-wrap items-center justify-center gap-3 text-sm text-slate-500">
              <span class="rounded-full bg-sky-600 px-3 py-1 text-xs font-medium text-white">
                Step {{ activeStudyIndex }}
              </span>
              <span class="rounded-full border border-slate-200 bg-white px-3 py-1 font-medium text-slate-700">
                {{ activeStudy.kicker }}
              </span>
            </div>

            <div class="mx-auto max-w-[60rem] space-y-3 text-center">
              <h3 class="font-display text-2xl font-medium leading-tight text-slate-950 sm:text-3xl">
                {{ activeStudy.title }}
              </h3>
              <p class="text-base leading-7 text-slate-600">
                {{ activeStudy.description }}
              </p>
            </div>

            <div class="grid gap-3 sm:grid-cols-3">
              <div v-for="metric in activeStudy.metrics" :key="`${activeStudy.id}-${metric.label}`" class="rounded-lg border border-slate-200 bg-slate-50/90 px-4 py-4">
                <div class="text-sm font-medium text-slate-600">{{ metric.label }}</div>
                <div class="mt-3 text-2xl font-medium text-slate-950">{{ metric.value }}</div>
              </div>
            </div>
          </div>
        </AppCard>

        <div class="grid gap-3 lg:grid-cols-4">
          <div v-for="item in benchmarkMatrix" :key="item.label" class="rounded-lg border border-slate-200 bg-white/82 p-4">
            <div class="text-sm font-medium text-slate-600">{{ item.label }}</div>
            <div class="mt-3 text-sm leading-7 text-slate-700">{{ item.value }}</div>
          </div>
        </div>
      </div>
    </section>

    <section id="taxonomy" class="scroll-mt-36 py-8 sm:py-10">
      <div class="mx-auto max-w-[1180px] space-y-5">
        <div class="mx-auto max-w-[54rem] space-y-2 text-center">
          <div class="text-sm font-medium text-slate-600">Taxonomy</div>
          <h2 class="font-display text-2xl font-medium leading-tight text-slate-950 sm:text-3xl xl:text-[2.2rem]">
            Ten collaborative failure modes across three phases.
          </h2>
          <p class="text-base leading-7 text-slate-600">
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
              class="rounded-full border px-4 py-2 text-sm font-medium transition"
              :class="switcherButtonClass(activePhaseId === phase.id)"
              @click="activePhaseId = phase.id"
            >
              {{ phase.label }}
            </button>
          </div>
        </div>

        <AppCard class="border-slate-200/80 bg-white/80" role="tabpanel" :aria-labelledby="`phase-tab-${activePhase.id}`">
          <div class="grid gap-5 p-5 sm:p-6 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
            <div class="space-y-3">
              <div class="inline-flex rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-sm font-medium text-teal-800">
                {{ activePhase.kicker }}
              </div>
              <h3 class="font-display text-2xl font-medium leading-tight text-slate-950 sm:text-3xl">
                {{ activePhase.title }}
              </h3>
              <p class="text-base leading-7 text-slate-600">{{ activePhase.summary }}</p>
            </div>

            <div class="grid gap-3 sm:grid-cols-2">
              <div v-for="mode in activePhase.modes" :key="mode.code" class="rounded-lg border border-slate-200 bg-slate-50/90 p-4">
                <div class="flex items-center justify-between gap-3">
                  <div class="text-sm font-medium text-slate-600">{{ mode.code }}</div>
                  <div class="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-slate-900 ring-1 ring-slate-200">
                    {{ mode.rate }}
                  </div>
                </div>
                <div class="mt-3 text-base font-medium text-slate-950">{{ mode.label }}</div>
              </div>
            </div>
          </div>
        </AppCard>
      </div>
    </section>

    <section id="results" class="scroll-mt-36 py-8 sm:py-10">
      <div class="mx-auto max-w-[1180px] space-y-5">
        <div class="flex flex-col gap-2 border-b border-slate-200 pb-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div class="text-sm font-medium text-slate-600">Results</div>
            <h2 class="font-display text-2xl font-medium leading-tight text-slate-950 sm:text-3xl">Key findings</h2>
          </div>
          <p class="max-w-[34rem] text-sm leading-6 text-slate-600">
            Phase-level failure rates and auditor validation from the MedAgentAudit study.
          </p>
        </div>

        <div class="overflow-hidden rounded-lg border border-slate-200 bg-white/76">
          <div
            v-for="(finding, index) in keyFindings"
            :key="finding.label"
            class="grid gap-2 border-slate-200 px-5 py-4 sm:grid-cols-[14rem_7rem_minmax(0,1fr)] sm:items-baseline"
            :class="index > 0 ? 'border-t' : ''"
          >
            <div class="text-sm font-medium text-slate-950">{{ finding.label }}</div>
            <div class="text-xl font-medium leading-tight text-slate-950">{{ finding.value }}</div>
            <div class="text-sm leading-6 text-slate-600">{{ finding.detail }}</div>
          </div>
        </div>
      </div>
    </section>

    <section id="annotation" class="scroll-mt-36 py-8 sm:py-10">
      <div class="mx-auto max-w-[1180px] space-y-5">
        <div class="flex flex-col gap-2 border-b border-slate-200 pb-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div class="text-sm font-medium text-slate-600">Annotation</div>
            <h2 class="font-display text-2xl font-medium leading-tight text-slate-950 sm:text-3xl">Evaluation tools</h2>
          </div>
          <p class="max-w-[34rem] text-sm leading-6 text-slate-600">
            Start from the annotation overview or jump directly into a review workflow.
          </p>
        </div>

        <div class="grid gap-3 md:grid-cols-2">
          <RouterLink
            v-for="card in annotationCards"
            :key="card.route"
            :to="card.route"
            class="group rounded-lg border border-slate-200/80 bg-white/76 p-5 transition hover:border-slate-400 hover:bg-white"
          >
            <div class="space-y-2">
              <div class="text-sm text-slate-500">{{ card.meta }}</div>
              <div class="text-xl font-medium text-slate-950">{{ card.title }}</div>
              <p class="text-sm leading-6 text-slate-600">{{ card.body }}</p>
              <div class="pt-1 text-sm font-medium text-slate-950 group-hover:underline">Open {{ card.title }}</div>
            </div>
          </RouterLink>
        </div>
      </div>
    </section>
      </div>
    </div>
  </AppShell>
</template>
