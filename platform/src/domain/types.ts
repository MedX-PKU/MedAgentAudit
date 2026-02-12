import type { TaxonomyKey } from './taxonomy'

export type DatasetName = string
export type FrameworkName = string

export type QuestionModality = 'text' | 'vqa'

export type QuestionBase = {
  caseId: string
  caseNumber?: number
  dataset: DatasetName
  framework: FrameworkName
  modality: QuestionModality
  question: string
  options?: string[]
  answer?: string
  predictedAnswer?: string
}

export type VqaInfo = {
  image?: { path: string; alt?: string }
}

export type OpenCodingCase = QuestionBase &
  VqaInfo & {
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
  taxonomyKey: TaxonomyKey
  context: string
  instructionText?: string
}

export type AuditCase = QuestionBase & VqaInfo & { items: AuditItem[]; collaborationLog?: unknown }

export type AuditAnnotation = {
  auditId: string
  caseId: string
  taxonomyKey: TaxonomyKey
  verdict: 'yes' | 'no'
  updatedAt: string
}
