import type { AuditCase } from '../../domain/types'

export const AUDIT_CASES: AuditCase[] = [
  {
    caseId: 'demo-audit-0001',
    dataset: 'MedQA',
    framework: 'ColaCare',
    modality: 'text',
    question:
      'A 70-year-old woman with fever and cough has a chest X-ray. The agent claims a right lower lobe consolidation, but the image note says left lower lobe involvement. Does the statement contain a factual hallucination?',
    options: ['Yes', 'No'],
    answer: 'Yes',
    items: [
      {
        auditId: 'demo-audit-0001__1.1.1',
        caseId: 'demo-audit-0001',
        taxonomyKey: '1.1.1',
        context:
          'Agent: "There is a right lower lobe consolidation consistent with pneumonia."\nProvided clinical note: "left lower lobe consolidation is present."\nQuestion: verify whether the agent description contradicts the given evidence.',
      },
      {
        auditId: 'demo-audit-0001__2.2.1',
        caseId: 'demo-audit-0001',
        taxonomyKey: '2.2.1',
        context:
          'Round 2: Agent B replies "I agree with Agent A. Inferior STEMI." without adding new evidence or correcting a flawed rationale.\nQuestion: is this discussion merely repetition of initial views?',
      },
    ],
  },
  {
    caseId: 'demo-audit-0002',
    dataset: 'VQA-RAD',
    framework: 'MAC',
    modality: 'vqa',
    question: 'Does the chest X-ray show signs of pleural effusion?',
    image: { path: '/data/images/demo/chest-xray.png', alt: 'Demo chest X-ray' },
    options: ['Yes', 'No'],
    answer: 'No',
    items: [
      {
        auditId: 'demo-audit-0002__1.2.1',
        caseId: 'demo-audit-0002',
        taxonomyKey: '1.2.1',
        context:
          'The agent answers based only on the textual question and does not reference any visual findings.\nQuestion: did the agent neglect required image modality?',
      },
    ],
  },
]
