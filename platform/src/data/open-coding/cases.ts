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

const fetchCaseIndex = async () => {
  const listResponse = await fetch(CASES_URL, { cache: 'no-store' })
  if (!listResponse.ok) throw new Error(`Failed to load ${CASES_URL}`)

  const contentType = listResponse.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) {
    throw new Error(`Expected a JSON case index at ${CASES_URL}. Add public/data/open-coding/index.json.`)
  }

  const files = (await listResponse.json()) as unknown
  if (!Array.isArray(files) || files.some((file) => typeof file !== 'string')) {
    throw new Error(`Invalid open-coding case index at ${CASES_URL}. Expected an array of JSONL file names.`)
  }

  return files as string[]
}

export const loadOpenCodingCases = async (): Promise<OpenCodingCase[]> => {
  if (cached) return cached
  const files = await fetchCaseIndex()

  const cases: OpenCodingCase[] = []
  for (const file of files) {
    const content = await fetchText(`${FILE_BASE_URL}/${file}`)
    cases.push(...parseOpenCodingJsonl(content, file))
  }
  // Ensure every case has a stable 1-based sequence number for display.
  cached = cases.map((c, idx) => ({ ...c, seq: c.seq ?? idx + 1 }))
  return cached
}
