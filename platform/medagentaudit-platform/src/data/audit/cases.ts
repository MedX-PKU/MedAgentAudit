import type { AuditCase } from '../../domain/types'

export const AUDIT_CASES: AuditCase[] = [
  {
    caseId: 'demo-audit-0001',
    dataset: 'MedQA',
    framework: 'ColaCare',
    modality: 'text',
    question: 'Demo audit case. Replace `src/data/audit/cases.ts` with real data.',
    options: ['A', 'B', 'C', 'D'],
    answer: 'A',
    items: [
      {
        auditId: 'demo-audit-0001__1.1.1',
        caseId: 'demo-audit-0001',
        taxonomyKey: '1.1.1',
        context: 'Show the minimal context for this failure mode here (not the full log).',
      },
    ],
  },
]

