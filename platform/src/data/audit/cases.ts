import type { AuditCase } from '../../domain/types'
import { parseAuditJsonl } from './jsonl'

const CASES_URL = '/data/audit/index.json'
const FILE_BASE_URL = '/data/audit'

let cached: AuditCase[] | null = null

const fetchText = async (url: string) => {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load ${url}`)
  return res.text()
}

export const loadAuditCases = async (): Promise<AuditCase[]> => {
  if (cached) return cached
  const listResponse = await fetch(CASES_URL)
  if (!listResponse.ok) throw new Error(`Failed to load ${CASES_URL}`)
  const files = (await listResponse.json()) as string[]

  const cases: AuditCase[] = []
  for (const file of files) {
    const content = await fetchText(`${FILE_BASE_URL}/${file}`)
    cases.push(...parseAuditJsonl(content, file))
  }

  // Assign a stable sequential ID so auditor assignment doesn't depend on qid formatting.
  // We encode it into `caseId` so the existing assignment & storage keys stay deterministic.
  cases.sort((a, b) => a.caseId.localeCompare(b.caseId))
  cases.forEach((c, idx) => {
    const seq = idx + 1
    c.caseNumber = seq
    c.caseId = `${c.caseId}__seq_${seq}`
    c.items = c.items.map((it) => ({
      ...it,
      caseId: c.caseId,
      auditId: `${c.caseId}__${it.taxonomyKey}`,
    }))
  })

  cached = cases
  return cases
}

export const AUDIT_CASES: AuditCase[] = []
