export const projectMeta = {
  name: 'MedAgentAudit',
  title: 'MedAgentAudit: Diagnosing and Quantifying Collaborative Failure Modes in Medical Multi-Agent Systems',
  titleLines: [
    'MedAgentAudit: Diagnosing and Quantifying Collaborative',
    'Failure Modes in Medical Multi-Agent Systems',
  ],
  authors: [
    { name: 'Yinghao Zhu', marks: ['1', '2', '*'] },
    { name: 'Lei Gu', marks: ['1', '*'] },
    { name: 'Zixiang Wang', marks: ['1', '*'] },
    { name: 'Haoran Sang', marks: ['1'] },
    { name: 'Dehao Sui', marks: ['1'] },
    { name: 'Wen Tang', marks: ['3'] },
    { name: 'Ewen Harrison', marks: ['4'] },
    { name: 'Junyi Gao', marks: ['4', '5'] },
    { name: 'Lequan Yu', marks: ['2', '#'] },
    { name: 'Liantao Ma', marks: ['1', '#'] },
  ],
  affiliations: [
    {
      mark: '1',
      text: 'National Engineering Research Center for Software Engineering, Peking University, Beijing, China, 100871',
    },
    {
      mark: '2',
      text: 'School of Computing and Data Science, The University of Hong Kong, Hong Kong SAR, China, 999077',
    },
    {
      mark: '3',
      text: 'Department of Nephrology, Peking University Third Hospital, Beijing, China, 100191',
    },
    {
      mark: '4',
      text: 'Centre for Medical Informatics, The University of Edinburgh, Edinburgh, UK, EH8 9YL',
    },
    {
      mark: '5',
      text: 'Health Data Research UK, London, UK',
    },
  ],
  notes: [
    { mark: '*', text: 'Equal contributions' },
    { mark: '#', text: 'Corresponding authors: Lequan Yu and Liantao Ma. Contact: yhzhu99@gmail.com' },
  ],
  links: [
    {
      label: 'Repository',
      href: 'https://github.com/MedX-PKU/MedAgentAudit',
      kind: 'primary' as const,
      icon: 'github' as const,
    },
    {
      label: 'Annotation',
      href: '/annotation',
      kind: 'secondary' as const,
      icon: null,
    },
  ],
}

export const sectionNav = [
  { id: 'overview', label: 'Overview' },
  { id: 'design', label: 'Study Design' },
  { id: 'taxonomy', label: 'Taxonomy' },
  { id: 'results', label: 'Results' },
  { id: 'annotation', label: 'Annotation' },
]

export const projectFacts = [
  {
    label: 'Logs analyzed',
    value: '3,600 logs',
  },
  {
    label: 'Failure modes',
    value: '10 modes',
  },
  {
    label: 'Audited cases',
    value: '14,400 cases',
  },
]

export const studyStages = [
  {
    id: 'baseline',
    shortLabel: 'Baseline',
    kicker: 'Final-answer comparison',
    title: 'Collaboration is compared against matched single-model baselines.',
    description:
      'The study first evaluates whether six medical MAS improve diagnostic accuracy over the same underlying LLM used as a single-model baseline across text QA and visual QA datasets.',
    metrics: [
      { label: 'Clinical test cases', value: '2,400' },
      { label: 'MAS frameworks', value: '6' },
      { label: 'Datasets', value: '6' },
    ],
  },
  {
    id: 'taxonomy',
    shortLabel: 'Taxonomy',
    kicker: 'Open coding',
    title: 'A phase-aligned failure taxonomy is built from collaboration traces.',
    description:
      'Open coding identifies recurrent collaborative failures, then double-blind expert annotation checks the codebook coverage and consistency on a holdout set.',
    metrics: [
      { label: 'Coded logs', value: '720' },
      { label: 'Holdout logs', value: '360' },
      { label: 'Cohen kappa', value: '0.76' },
    ],
  },
  {
    id: 'audit',
    shortLabel: 'Audit',
    kicker: 'Probe-based evaluation',
    title: 'Auditor probes locate failures without changing the MAS workflow.',
    description:
      'An automated auditor reads phase-matched context at predefined interaction steps to quantify where failures enter, whether they persist, and how they affect later decisions.',
    metrics: [
      { label: 'Audited cases', value: '14,400' },
      { label: 'Validation set', value: '400' },
      { label: 'Macro F1', value: '0.845' },
    ],
  },
]

export const phaseDeck = [
  {
    id: 'phase-1',
    label: 'Phase 1',
    kicker: 'Task comprehension',
    title: 'Input is misperceived or missed before collaboration begins.',
    summary:
      'Factual hallucinations and modality neglect can distort the first interpretation of the case, and later discussion often builds on that flawed reading instead of re-checking the source input.',
    modes: [
      { code: 'F-1.1.1', label: 'Factual hallucinations', rate: '16.63%' },
      { code: 'F-1.2.1', label: 'Modality neglect', rate: '4.59%' },
    ],
  },
  {
    id: 'phase-2',
    label: 'Phase 2',
    kicker: 'Collaborative discussion',
    title: 'Discussion repeats prior views and can leave conflicts unresolved.',
    summary:
      'Agents often preserve first-round judgments, fail to activate role-specific expertise, or continue despite mutually incompatible clinical claims.',
    modes: [
      { code: 'F-2.1.1', label: 'Role-task mismatch', rate: '22.71%' },
      { code: 'F-2.1.2', label: 'Specialist knowledge inactive', rate: '42.73%' },
      { code: 'F-2.2.1', label: 'Repetition of initial views', rate: '98.42%' },
      { code: 'F-2.2.2', label: 'Unresolved conflicts', rate: '9.33%' },
    ],
  },
  {
    id: 'phase-3',
    label: 'Phase 3',
    kicker: 'Synthesis and decision',
    title: 'Final synthesis favors authority, majority, or prior conclusions.',
    summary:
      'Decision steps can treat role labels, majority answers, or previous summaries as substitutes for checking whether the clinical reasoning is supported by the case facts.',
    modes: [
      { code: 'F-3.1.1', label: 'Correct minority suppressed', rate: '5.11%' },
      { code: 'F-3.1.2', label: 'Authority bias', rate: '28.76%' },
      { code: 'F-3.1.3', label: 'Contradiction neglect', rate: '5.48%' },
      { code: 'F-3.2.1', label: 'Cross-round self-contradiction', rate: '18.53%' },
    ],
  },
]

export const keyFindings = [
  {
    label: 'Task comprehension',
    value: '16.63%',
    detail: 'Factual hallucinations; modality neglect reaches 4.59%.',
  },
  {
    label: 'Collaborative discussion',
    value: '98.42%',
    detail: 'Repetition of initial views dominates the discussion phase.',
  },
  {
    label: 'Synthesis and decision',
    value: '28.76%',
    detail: 'Authority bias is the most frequent decision-stage failure.',
  },
  {
    label: 'Auditor validation',
    value: '0.845',
    detail: 'Macro F1 on the human-annotated validation set.',
  },
]

export const benchmarkMatrix = [
  { label: 'Text QA datasets', value: 'MedQA, PubMedQA, MedXpertQA' },
  { label: 'Visual QA datasets', value: 'PathVQA, VQA-RAD, SLAKE' },
  { label: 'MAS frameworks', value: 'ColaCare, MAC, HealthcareAgent, MDAgents, MedAgents, ReConcile' },
  { label: 'Underlying LLMs', value: 'DeepSeek-V3.2, GPT-5.2, Gemini-3-Flash, GLM-4.6V, Qwen-3, Qwen-3VL' },
]

export const annotationCards = [
  {
    title: 'Open-coding',
    route: '/annotation/open-coding',
    body: 'Code complete collaboration traces using the 10-mode taxonomy and capture novel failure patterns when the codebook is insufficient.',
    meta: 'Open coding',
    cta: 'Start Open-coding',
    highlights: ['Full collaboration trace', '10-mode taxonomy', 'Novel failure note'],
  },
  {
    title: 'Audit',
    route: '/annotation/audit',
    body: 'Validate targeted failure-mode judgments with phase-matched context, deterministic assignments, and local JSON export.',
    meta: 'Audit validation',
    cta: 'Start Audit',
    highlights: ['Mode-level verdict', 'Phase-matched context', 'Auditor assignment'],
  },
]

export const annotationOverviewStats = [
  {
    label: 'Open-coding',
    value: '10 modes',
    note: 'Complete trace review with optional novel failure capture.',
  },
  {
    label: 'Audit validation',
    value: 'Yes / No',
    note: 'Focused failure-mode verification against supporting context.',
  },
  {
    label: 'Export',
    value: 'JSON',
    note: 'Browser-local annotation state with explicit export files.',
  },
]

export const annotationWorkflowSteps = [
  {
    label: 'Choose',
    title: 'Select the review task',
    body: 'Open-coding builds taxonomy evidence; audit validation checks individual failure-mode claims.',
  },
  {
    label: 'Annotate',
    title: 'Work through assigned cases',
    body: 'Each tool keeps case context, collaboration logs, progress, and decision controls in one workspace.',
  },
  {
    label: 'Export',
    title: 'Download structured labels',
    body: 'Completed annotations export as schema-tagged JSON for downstream analysis.',
  },
]
