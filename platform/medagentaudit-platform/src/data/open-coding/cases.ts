import type { OpenCodingCase } from '../../domain/types'

export const OPEN_CODING_CASES: OpenCodingCase[] = [
  {
    caseId: 'demo-open-0001',
    dataset: 'MedQA',
    framework: 'ColaCare',
    modality: 'text',
    question:
      'A 65-year-old man presents with acute chest pain radiating to the left arm. ECG shows ST elevation in leads II, III, aVF. What is the most likely diagnosis?',
    options: ['A. Anterior STEMI', 'B. Inferior STEMI', 'C. Pericarditis', 'D. Aortic dissection'],
    answer: 'A',
    collaborationLog: {
      case_history: {
        rounds: [
          {
            round: 1,
            opinions: [
              {
                agent_id: 'doctor_1',
                specialty: 'Internal Medicine',
                log: { parsed_output: { answer: 'B', explanation: 'ST elevation in II/III/aVF indicates inferior MI.' } },
              },
              {
                agent_id: 'doctor_2',
                specialty: 'Surgery',
                log: { parsed_output: { answer: 'D', explanation: 'Chest pain radiating could indicate aortic dissection.' } },
              },
              {
                agent_id: 'doctor_3',
                specialty: 'Radiology',
                log: { parsed_output: { answer: 'B', explanation: 'Inferior wall involvement fits II/III/aVF.' } },
              },
            ],
            synthesis: { parsed_output: { answer: 'B', explanation: 'Majority votes for inferior STEMI.' } },
            reviews: [
              {
                agent_id: 'doctor_2',
                specialty: 'Surgery',
                log: {
                  parsed_output: { agree: false, answer: 'D', reason: 'Consider dissection; ECG may be misleading.' },
                },
              },
            ],
            decision: { parsed_output: { answer: 'B', explanation: 'Finalize as inferior STEMI.' } },
          },
        ],
      },
    },
  },
  {
    caseId: 'demo-open-0002',
    dataset: 'VQA-RAD',
    framework: 'MedAgent',
    modality: 'vqa',
    question: 'Is there evidence of pneumonia in the provided chest X-ray?',
    image: { path: '/data/images/demo/chest-xray.png', alt: 'Demo chest X-ray' },
    options: ['Yes', 'No'],
    answer: 'Yes',
    collaborationLog: {
      case_history: {
        rounds: [
          {
            round: 1,
            opinions: [
              {
                agent_id: 'doctor_1',
                specialty: 'Radiology',
                log: { parsed_output: { answer: 'No', explanation: 'Lungs appear clear; no consolidation.' } },
              },
              {
                agent_id: 'doctor_2',
                specialty: 'Internal Medicine',
                log: { parsed_output: { answer: 'Yes', explanation: 'Clinical suspicion; mild opacity possibly present.' } },
              },
            ],
            synthesis: { parsed_output: { answer: 'No', explanation: 'Radiology view prioritized.' } },
            decision: { parsed_output: { answer: 'No', explanation: 'No pneumonia.' } },
          },
        ],
      },
    },
  },
]
