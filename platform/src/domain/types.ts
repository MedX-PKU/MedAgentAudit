import type { TaxonomyKey } from './taxonomy'

export type DatasetName = string
export type FrameworkName = string

export type QuestionModality = 'text' | 'vqa'

export type QuestionBase = {
  caseId: string
  dataset: DatasetName
  framework: FrameworkName
  modality: QuestionModality
  question: string
  options?: string[]
  answer?: string
}

export type VqaInfo = {
  image?: { path: string; alt?: string }
}

export type OpenCodingCase = QuestionBase &
  VqaInfo & {
    collaborationLog: unknown
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
}

export type AuditCase = QuestionBase & VqaInfo & { items: AuditItem[] }

export type AuditAnnotation = {
  auditId: string
  caseId: string
  taxonomyKey: TaxonomyKey
  verdict: 'yes' | 'no'
  updatedAt: string
}
