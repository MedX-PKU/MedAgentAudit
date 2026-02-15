import type { AuditAnnotation } from '../../../domain/types'
import { readJson, writeJson } from '../../../lib/storage'
import type { AuditorId } from './auditAssignment'

const keyFor = (auditorId: AuditorId) => `medagentaudit:audit:auditor:${auditorId}`
const keyForCase = (seq: number | undefined, caseId: string) => `case_${seq ?? 'unknown'}_${caseId}`

export const loadAuditMap = (auditorId: AuditorId): Record<string, AuditAnnotation> => {
  const raw = readJson<Record<string, AuditAnnotation>>(keyFor(auditorId), {})
  const result: Record<string, AuditAnnotation> = {}
  for (const [k, v] of Object.entries(raw)) {
    const normalized = { ...v, auditorId: v.auditorId ?? auditorId }
    // Back-compat: previous keys were various auditId formats. Normalize to `case_{seq}_{caseId}` when possible.
    const nextKey = k.startsWith('case_') ? k : keyForCase(normalized.seq, normalized.caseId)
    result[nextKey] = { ...normalized, auditId: nextKey }
  }
  return result
}

export const saveAudit = (auditorId: AuditorId, annotation: AuditAnnotation) => {
  const map = loadAuditMap(auditorId)
  const stored = { ...annotation, auditorId }
  const key = keyForCase(stored.seq, stored.caseId)
  map[key] = { ...stored, auditId: key }
  writeJson(keyFor(auditorId), map)
}
