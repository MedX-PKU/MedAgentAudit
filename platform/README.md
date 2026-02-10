# MedAgentAudit Platform

Browser-only annotation platform for the MedAgentAudit project.

## Routes

- `/` project landing page
- `/annotation/open-coding` open-coding UI (full log + taxonomy multi-select)
- `/annotation/audit` audit UI (mode-level yes/no + deterministic assignment)

## Data conventions (static JS/TS, no backend)

- Open-coding cases: `src/data/open-coding/cases.ts`
- Audit cases: `src/data/audit/cases.ts`
- VQA images: `public/data/images/...` and reference via `image.path`

## Local persistence

- Open-coding saved under `localStorage` key `medagentaudit:open-coding:<name>`
- Audit saved under `localStorage` key `medagentaudit:audit:auditor:<id>`

## Export

- Open-coding: `{name}_opencoding.json`
- Audit: `Auditor_<id>_audit.json`

## Dev

```bash
npm i
npm run dev
```

## Build

```bash
npm run build
```
