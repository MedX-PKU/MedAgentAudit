<script setup lang="ts">
import AppButton from '../../components/ui/AppButton.vue'
import AppCard from '../../components/ui/AppCard.vue'
import AppShell from '../../components/layout/AppShell.vue'
</script>

<template>
  <AppShell title="MedAgentAudit">
    <div class="space-y-8">
      <section class="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
        <div class="text-sm font-semibold text-blue-700">Paper project page</div>
        <h1 class="mt-2 text-3xl font-semibold tracking-tight text-slate-900">
          Diagnosing and Quantifying Collaborative Failure Modes in Medical Multi-Agent Systems
        </h1>
        <p class="mt-3 max-w-3xl text-slate-600">
          This page hosts the public project landing content and a browser-only annotation tool (open-coding + audit).
        </p>
        <div class="mt-6 flex flex-wrap gap-3">
          <RouterLink to="/annotation/open-coding">
            <AppButton>Go to Annotation</AppButton>
          </RouterLink>
          <a href="https://github.com" target="_blank" rel="noreferrer">
            <AppButton variant="secondary">Repository</AppButton>
          </a>
        </div>
      </section>

      <section class="grid gap-4 md:grid-cols-2">
        <AppCard class="p-5">
          <div class="text-sm font-semibold text-slate-900">Open-coding</div>
          <p class="mt-1 text-sm text-slate-600">
            Review the full collaboration log and label failure modes using the taxonomy (multi-select). Optionally add a
            novel failure mode in free text.
          </p>
        </AppCard>
        <AppCard class="p-5">
          <div class="text-sm font-semibold text-slate-900">Audit (mode-level)</div>
          <p class="mt-1 text-sm text-slate-600">
            Review minimal context for a specific failure mode and answer yes/no. Cases are assigned deterministically so
            each item is reviewed by 3 auditors.
          </p>
        </AppCard>
      </section>
    </div>
  </AppShell>
</template>

