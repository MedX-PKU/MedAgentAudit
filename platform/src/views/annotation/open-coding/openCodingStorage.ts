import type { OpenCodingAnnotation } from '../../../domain/types'
import { readJson, writeJson } from '../../../lib/storage'

const keyFor = (annotatorName: string) => `medagentaudit:open-coding:${annotatorName}`
const keyForCase = (seq: number | undefined, caseId: string) =>
  `case_${seq ?? 'unknown'}_${caseId}`

export const loadOpenCodingMap = (annotatorName: string): Record<string, OpenCodingAnnotation> => {
  const raw = readJson<Record<string, OpenCodingAnnotation>>(keyFor(annotatorName), {})
  const result: Record<string, OpenCodingAnnotation> = {}

  for (const [storedKey, ann] of Object.entries(raw)) {
    // Back-compat: previously keyed by raw caseId.
    if (storedKey.startsWith('case_')) {
      result[storedKey] = ann
      continue
    }
    const nextKey = keyForCase(ann.seq, ann.caseId)
    result[nextKey] = ann
  }

  return result
}

export const saveOpenCoding = (annotatorName: string, annotation: OpenCodingAnnotation) => {
  const map = loadOpenCodingMap(annotatorName)
  map[keyForCase(annotation.seq, annotation.caseId)] = annotation
  writeJson(keyFor(annotatorName), map)
}
