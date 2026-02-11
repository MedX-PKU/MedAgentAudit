import type { AuditCase, AuditItem } from '../../../domain/types'
import { fnv1a32 } from '../../../lib/seeded'

export type AuditorId = 1 | 2 | 3 | 4 | 5 | 6

export const AUDITOR_IDS: AuditorId[] = [1, 2, 3, 4, 5, 6]

const pick3Of6 = (caseId: string): AuditorId[] => {
  const scored = AUDITOR_IDS.map((id) => ({ id, score: fnv1a32(`${caseId}::auditor::${id}`) }))
  scored.sort((a, b) => a.score - b.score)
  return scored.slice(0, 3).map((s) => s.id)
}

export const isAssigned = (auditor: AuditorId, caseId: string): boolean => {
  return pick3Of6(caseId).includes(auditor)
}

export const assignedAuditItems = (auditor: AuditorId, cases: readonly AuditCase[]): AuditItem[] => {
  const items: AuditItem[] = []
  for (const c of cases) {
    if (!isAssigned(auditor, c.caseId)) continue
    for (const item of c.items) items.push(item)
  }
  return items
}

