import type { AuditAnnotation } from '../../../domain/types'
import { readJson, writeJson } from '../../../lib/storage'
import type { AuditorId } from './auditAssignment'

const keyFor = (auditorId: AuditorId) => `medagentaudit:audit:auditor:${auditorId}`

export const loadAuditMap = (auditorId: AuditorId): Record<string, AuditAnnotation> => {
  return readJson<Record<string, AuditAnnotation>>(keyFor(auditorId), {})
}

export const saveAudit = (auditorId: AuditorId, annotation: AuditAnnotation) => {
  const map = loadAuditMap(auditorId)
  map[annotation.auditId] = annotation
  writeJson(keyFor(auditorId), map)
}

