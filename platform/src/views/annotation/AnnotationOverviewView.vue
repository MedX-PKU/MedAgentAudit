<script setup lang="ts">
import AppCard from '../../components/ui/AppCard.vue'
import { annotationCards, annotationOverviewStats, annotationWorkflowSteps } from '../../content/site'

const primaryRoute = annotationCards[0]?.route ?? '/annotation/open-coding'
const secondaryRoute = annotationCards[1]?.route ?? '/annotation/audit'
</script>

<template>
  <div class="mx-auto max-w-[1180px] space-y-8 py-8 sm:py-10 xl:py-12">
    <section class="grid gap-7 lg:grid-cols-[minmax(0,0.98fr)_minmax(320px,0.72fr)] lg:items-end">
      <div class="space-y-5">
        <div class="text-sm font-medium text-slate-600">Annotation Workspace</div>
        <h1 class="font-display max-w-[56rem] text-[2rem] font-medium leading-[1.13] text-slate-950 sm:text-[2.55rem] lg:text-[2.72rem]">
          Choose the right review workflow for MedAgentAudit evidence.
        </h1>
        <p class="max-w-[45rem] text-base leading-7 text-slate-600">
          Start with open-coding when reviewing complete collaboration traces, or use audit validation when checking one failure-mode judgment against phase-matched context.
        </p>
        <div class="flex flex-wrap gap-3">
          <RouterLink
            :to="primaryRoute"
            class="inline-flex items-center justify-center rounded-full bg-slate-950 px-4 py-2 text-sm font-semibold text-white shadow-[0_16px_30px_rgba(15,23,42,0.14)] transition hover:bg-slate-800 focus:outline-none focus:ring-4 focus:ring-blue-500/20"
          >
            Start Open-coding
          </RouterLink>
          <RouterLink
            :to="secondaryRoute"
            class="inline-flex items-center justify-center rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-900 ring-1 ring-slate-200 transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-blue-500/20"
          >
            Start Audit
          </RouterLink>
        </div>
      </div>

      <AppCard class="overflow-hidden border-slate-200/80 bg-white/80">
        <div class="border-b border-slate-200 px-5 py-4">
          <div class="text-sm font-medium text-slate-600">Workspace Summary</div>
          <div class="mt-1 text-2xl font-medium leading-tight text-slate-950">Two complementary annotation tools</div>
        </div>
        <div class="divide-y divide-slate-200">
          <div v-for="card in annotationCards" :key="card.route" class="px-5 py-4">
            <div class="flex items-start justify-between gap-4">
              <div>
                <div class="text-sm text-slate-500">{{ card.meta }}</div>
                <div class="mt-1 text-lg font-medium text-slate-950">{{ card.title }}</div>
              </div>
              <RouterLink
                :to="card.route"
                class="shrink-0 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-900 transition hover:border-slate-400 hover:bg-slate-50"
              >
                Open
              </RouterLink>
            </div>
          </div>
        </div>
      </AppCard>
    </section>

    <section class="grid border-y border-slate-200 bg-white/54 sm:grid-cols-3">
      <div
        v-for="stat in annotationOverviewStats"
        :key="stat.label"
        class="border-slate-200 px-5 py-4 sm:border-l first:sm:border-l-0"
      >
        <div class="text-sm text-slate-500">{{ stat.label }}</div>
        <div class="mt-1 text-[1.65rem] font-medium leading-tight text-slate-950">{{ stat.value }}</div>
        <p class="mt-2 text-sm leading-6 text-slate-600">{{ stat.note }}</p>
      </div>
    </section>

    <section class="space-y-5">
      <div class="flex flex-col gap-2 border-b border-slate-200 pb-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div class="text-sm font-medium text-slate-600">Workflows</div>
          <h2 class="font-display text-2xl font-medium leading-tight text-slate-950 sm:text-3xl">Evaluation tools</h2>
        </div>
        <p class="max-w-[34rem] text-sm leading-6 text-slate-600">
          Pick the task that matches the evidence being reviewed.
        </p>
      </div>

      <div class="grid gap-3 lg:grid-cols-2">
        <RouterLink
          v-for="card in annotationCards"
          :key="card.route"
          :to="card.route"
          class="group rounded-lg border border-slate-200/80 bg-white/76 p-5 transition hover:border-slate-400 hover:bg-white"
        >
          <div class="flex h-full flex-col justify-between gap-5">
            <div class="space-y-3">
              <div class="inline-flex rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-sm font-medium text-sky-900">
                {{ card.meta }}
              </div>
              <div class="space-y-2">
                <h3 class="font-display text-2xl font-medium leading-tight text-slate-950">{{ card.title }}</h3>
                <p class="text-sm leading-6 text-slate-600">{{ card.body }}</p>
              </div>
              <div class="flex flex-wrap gap-2">
                <span
                  v-for="item in card.highlights"
                  :key="`${card.route}-${item}`"
                  class="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700"
                >
                  {{ item }}
                </span>
              </div>
            </div>
            <div class="text-sm font-medium text-slate-950 group-hover:underline">{{ card.cta }}</div>
          </div>
        </RouterLink>
      </div>
    </section>

    <section class="space-y-5">
      <div class="flex flex-col gap-2 border-b border-slate-200 pb-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div class="text-sm font-medium text-slate-600">Process</div>
          <h2 class="font-display text-2xl font-medium leading-tight text-slate-950 sm:text-3xl">Annotation flow</h2>
        </div>
        <p class="max-w-[34rem] text-sm leading-6 text-slate-600">
          Both tools keep progress in the browser and export structured labels.
        </p>
      </div>

      <div class="grid gap-3 md:grid-cols-3">
        <AppCard v-for="step in annotationWorkflowSteps" :key="step.label" class="bg-white/76 p-5">
          <div class="space-y-3">
            <div class="inline-flex rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-sm font-medium text-teal-800">
              {{ step.label }}
            </div>
            <h3 class="text-lg font-medium leading-tight text-slate-950">{{ step.title }}</h3>
            <p class="text-sm leading-6 text-slate-600">{{ step.body }}</p>
          </div>
        </AppCard>
      </div>
    </section>
  </div>
</template>
