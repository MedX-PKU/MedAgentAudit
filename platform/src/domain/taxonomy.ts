export type TaxonomyKey =
  | '1.1.1'
  | '1.2.1'
  | '2.1.1'
  | '2.1.2'
  | '2.2.1'
  | '2.2.2'
  | '3.1.1'
  | '3.1.2'
  | '3.1.3'
  | '3.2.1'

export type TaxonomyItem = { key: TaxonomyKey; title: string; short: string; phase: 'I' | 'II' | 'III' }

export const TAXONOMY: TaxonomyItem[] = [
  { key: '1.1.1', phase: 'I', title: 'Factual Hallucinations', short: 'Factual hallucination during input interpretation' },
  { key: '1.2.1', phase: 'I', title: 'Modality Neglect / Misinterpretation', short: 'Ignores or misinterprets required modality / task intent' },

  { key: '2.1.1', phase: 'II', title: 'Role–Task Mismatch', short: 'Assigned roles do not match clinical task' },
  { key: '2.1.2', phase: 'II', title: 'Failure to Activate Specialist Knowledge', short: 'Role setting fails to elicit specialist reasoning' },
  { key: '2.2.1', phase: 'II', title: 'Repetition of Initial Views', short: 'Discussion repeats initial views without new info' },
  { key: '2.2.2', phase: 'II', title: 'Unresolved Conflicts', short: 'Contradictory viewpoints not resolved' },

  { key: '3.1.1', phase: 'III', title: 'Suppression of Correct Minority', short: 'Incorrect consensus suppresses correct minority view' },
  { key: '3.1.2', phase: 'III', title: 'Authority Bias', short: 'Decision distorted by role/format authority rather than evidence' },
  { key: '3.1.3', phase: 'III', title: 'Neglect of Contradictions', short: 'Integrates views without checking reasoning contradictions' },
  { key: '3.2.1', phase: 'III', title: 'Self-Contradiction Across Rounds', short: 'Decision flips across rounds without new evidence' },
]

