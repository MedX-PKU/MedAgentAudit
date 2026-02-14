import type { AuditCase, AuditItem } from '../../../domain/types'

export type AuditorId = 1 | 2 | 3 | 4 | 5 | 6

export const AUDITOR_IDS: AuditorId[] = [1, 2, 3, 4, 5, 6]

/**
 * Check if a case with given seq is assigned to this auditor.
 * Each of 400 cases (seq 1-400) is assigned to exactly 3 auditors.
 */
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

/** Get case by seq (primary key). */
export const getCaseBySeq = (cases: readonly AuditCase[], seq: number): AuditCase | undefined => {
  return cases.find((c) => c.seq === seq)
}

/** Filter cases by seq range [min, max]. */
export const filterCasesBySeqRange = (
  cases: readonly AuditCase[],
  minSeq: number,
  maxSeq: number,
): AuditCase[] => {
  return cases.filter((c) => c.seq != null && c.seq >= minSeq && c.seq <= maxSeq)
}
