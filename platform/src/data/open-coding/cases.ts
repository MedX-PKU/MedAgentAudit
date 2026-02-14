import type { OpenCodingCase } from '../../domain/types'
import { parseOpenCodingJsonl } from './jsonl'

const CASES_URL = '/data/open-coding/index.json'
const FILE_BASE_URL = '/data/open-coding'

let cached: OpenCodingCase[] | null = null

const fetchText = async (url: string) => {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load ${url}`)
  return res.text()
}

export const loadOpenCodingCases = async (): Promise<OpenCodingCase[]> => {
  if (cached) return cached
  const listResponse = await fetch(CASES_URL)
  if (!listResponse.ok) throw new Error(`Failed to load ${CASES_URL}`)
  const files = (await listResponse.json()) as string[]

  const cases: OpenCodingCase[] = []
  for (const file of files) {
    const content = await fetchText(`${FILE_BASE_URL}/${file}`)
    cases.push(...parseOpenCodingJsonl(content, file))
  }
  // Ensure every case has a stable 1-based sequence number for display.
  cached = cases.map((c, idx) => ({ ...c, seq: c.seq ?? idx + 1 }))
  return cached
}
