# Data convention

All annotation data are shipped as static JS/TS modules and bundled by Vite.

## Open-coding data

- Location: `public/data/open-coding/*.jsonl`
- Parser: `src/data/open-coding/jsonl.ts`
- Loader: `loadOpenCodingCases(): Promise<OpenCodingCase[]>`

## Audit data (mode-level)

- Location: `public/data/audit/*.jsonl`
- Loader: `loadAuditCases(): Promise<AuditCase[]>`

## Images (VQA)

Place images under `public/data/...` and reference via:

- `image.path`: `/data/<...>`
