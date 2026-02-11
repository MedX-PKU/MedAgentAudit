import type { OpenCodingAnnotation } from '../../../domain/types'
import { readJson, writeJson } from '../../../lib/storage'

const keyFor = (annotatorName: string) => `medagentaudit:open-coding:${annotatorName}`

export const loadOpenCodingMap = (annotatorName: string): Record<string, OpenCodingAnnotation> => {
  return readJson<Record<string, OpenCodingAnnotation>>(keyFor(annotatorName), {})
}

export const saveOpenCoding = (annotatorName: string, annotation: OpenCodingAnnotation) => {
  const map = loadOpenCodingMap(annotatorName)
  map[annotation.caseId] = annotation
  writeJson(keyFor(annotatorName), map)
}

