# Data convention

All annotation data are shipped as static JS/TS modules and bundled by Vite.

## Open-coding data

- Location: `src/data/open-coding/cases.ts`
- Export: `OPEN_CODING_CASES: OpenCodingCase[]`

## Audit data (mode-level)

- Location: `src/data/audit/cases.ts`
- Export: `AUDIT_CASES: AuditCase[]`

## Images (VQA)

Place images under `public/data/images/...` and reference via:

- `image.path`: `/data/images/<...>`

