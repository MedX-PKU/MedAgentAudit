import type { AuditCase, AuditItem } from '../../../domain/types'
export type AuditorId = 1 | 2 | 3 | 4 | 5 | 6

export const AUDITOR_IDS: AuditorId[] = [1, 2, 3, 4, 5, 6]

export const isAssigned = (auditor: AuditorId, seq?: number): boolean => {
  if (!Number.isFinite(seq)) return false
  const n = seq as number

  const mod = ((n % 6) + 6) % 6
  const a1 = (mod + 1) as AuditorId
  const a2 = (((mod + 1) % 6) + 1) as AuditorId
  const a3 = (((mod + 2) % 6) + 1) as AuditorId

  return auditor === a1 || auditor === a2 || auditor === a3
}

export const assignedAuditItems = (auditor: AuditorId, cases: readonly AuditCase[]): AuditItem[] => {
  const items: AuditItem[] = []
  for (const c of cases) {
    if (!isAssigned(auditor, c.seq)) continue
    for (const item of c.items) items.push(item)
  }
  return items
}

export const assignedAuditCases = (auditor: AuditorId, cases: readonly AuditCase[]): AuditCase[] => {
  return cases.filter((c) => isAssigned(auditor, c.seq))
}
