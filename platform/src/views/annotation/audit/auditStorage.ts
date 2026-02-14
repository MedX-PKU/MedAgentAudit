import type { AuditAnnotation } from '../../../domain/types'
import { readJson, writeJson } from '../../../lib/storage'
import type { AuditorId } from './auditAssignment'

const keyFor = (auditorId: AuditorId) => `medagentaudit:audit:auditor:${auditorId}`

export const loadAuditMap = (auditorId: AuditorId): Record<string, AuditAnnotation> => {
  const raw = readJson<Record<string, AuditAnnotation>>(keyFor(auditorId), {})
  const result: Record<string, AuditAnnotation> = {}
  for (const [k, v] of Object.entries(raw)) {
    result[k] = { ...v, auditorId: v.auditorId ?? auditorId }
  }
  return result
}

export const saveAudit = (auditorId: AuditorId, annotation: AuditAnnotation) => {
  const map = loadAuditMap(auditorId)
  const stored = { ...annotation, auditorId }
  map[annotation.auditId] = stored
  writeJson(keyFor(auditorId), map)
}

