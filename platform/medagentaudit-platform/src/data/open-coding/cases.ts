import type { OpenCodingCase } from '../../domain/types'

export const OPEN_CODING_CASES: OpenCodingCase[] = [
  {
    caseId: 'demo-open-0001',
    dataset: 'MedQA',
    framework: 'ColaCare',
    modality: 'text',
    question: 'Demo case. Replace `src/data/open-coding/cases.ts` with real data.',
    options: ['A', 'B', 'C', 'D'],
    answer: 'A',
    collaborationLog: {
      note: 'Paste the original collaboration log object here.',
    },
  },
]

