import type { TaxonomyKey } from './taxonomy'

export type DatasetName = string
export type FrameworkName = string

export type QuestionModality = 'text' | 'vqa'

export type QuestionBase = {
  caseId: string
  seq?: number
  dataset: DatasetName
  framework: FrameworkName
  /** LLM identifier (e.g. gpt-5.2, qwen3-vl-8b-thinking); used for dedup when same case has different LLMs */
  llm?: string
  modality: QuestionModality
  question: string
  questionType?: string
  options?: string[]
  answer?: string
  predictedAnswer?: string
}

export type VqaInfo = {
  image?: { path: string; alt?: string }
}

export type OpenCodingCase = QuestionBase &
  VqaInfo & {
    collectionText?: string
    failureModeDefinitionMapping?: Record<
      string,
      { name?: string; definition?: string; human_eval_instruction?: string; humanEvalInstruction?: string }
    >
    collaborationLog: unknown
    instructionText?: string
  }

export type OpenCodingAnnotation = {
  caseId: string
  taxonomy: TaxonomyKey[]
  novelFailureMode?: string
  updatedAt: string
}

export type AuditItem = {
  auditId: string
  caseId: string
  /** Stable 1-based sequence (1-400), used as primary key for assignment and aggregation */
  seq?: number
  taxonomyKey: TaxonomyKey
  context: string
  instructionText?: string
}

export type AuditCase = QuestionBase & VqaInfo & { items: AuditItem[]; collaborationLog?: unknown }

export type AuditAnnotation = {
  auditId: string
  caseId: string
  /** Stable 1-based sequence (1-400), primary key for aggregation */
  seq?: number
  /** Auditor who made this annotation (1-6); enables multi-auditor per case */
  auditorId?: number
  taxonomyKey: TaxonomyKey
  verdict: 'yes' | 'no'
  updatedAt: string
}
