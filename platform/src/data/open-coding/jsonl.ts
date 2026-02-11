import type { OpenCodingCase } from '../../domain/types'

type JsonlRecord = {
  qid?: string
  case_id?: string
  caseId?: string
  image_path?: string | null
  question_description?: string
  collaboration_text?: string
  collaboration_log?: unknown
}

type ParsedOutput = { answer?: string; explanation?: string }
type Opinion = { agent_id: string; specialty?: string; log: { parsed_output: ParsedOutput } }
type Review = { agent_id: string; specialty?: string; log: { parsed_output: { agree?: boolean; answer?: string; reason?: string; explanation?: string } } }
type Round = { round: number; opinions: Opinion[]; synthesis?: { parsed_output: ParsedOutput }; reviews?: Review[]; decision?: { parsed_output: ParsedOutput } }
type ParsedCollaboration = { raw: string; case_history: { rounds: Round[] } }

const slugify = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

const normalizeFramework = (value: string) => {
  const normalized = value.trim().toLowerCase()
  if (normalized === 'medagent') return 'MedAgent'
  if (normalized === 'healthcareagent') return 'HealthcareAgent'
  if (normalized === 'mdagents') return 'MDAgents'
  if (normalized === 'colacare') return 'ColaCare'
  if (normalized === 'reconcile') return 'Reconcile'
  if (normalized === 'mac') return 'MAC'
  return value
    .replace(/[-_]+/g, ' ')
    .replace(/[\s]+(.)?/g, (_match: string, chr?: string) => (chr ? chr.toUpperCase() : ''))
}

const normalizeDataset = (value: string) => {
  const trimmed = value.trim()
  if (!trimmed) return 'Unknown'
  return trimmed
    .split(/[-_]+/)
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join('-')
}

const parseOptions = (text: string) => {
  const lines = text.split(/\r?\n/)
  const options: string[] = []
  let inOptions = false
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) continue
    if (line.startsWith('Options:')) {
      inOptions = true
      continue
    }
    if (!inOptions) continue
    const match = line.match(/^([A-Z]|\d+)\s*[:.)]\s*(.+)$/)
    if (match) {
      const label = match[1]
      const content = match[2]
      options.push(`${label}. ${content}`.trim())
      continue
    }
    if (/^The ground truth answer is:/i.test(line)) break
    if (options.length > 0) {
      options[options.length - 1] = `${options[options.length - 1]} ${line}`.trim()
    }
  }
  return options.length ? options : undefined
}

const parseQuestion = (text: string) => {
  const match = text.match(/The question is:(.*?)(?:\nThis question has|\nOptions:|\nThe ground truth answer is:)/s)
  if (!match) return text.trim()
  return match[1].trim().replace(/"\.?\s*$/, '')
}

const parseAnswer = (text: string) => {
  const match = text.match(/The ground truth answer is:\s*([A-Z0-9]+)\b/i)
  return match ? match[1].trim() : undefined
}

const resolveImagePath = (imagePath?: string | null) => {
  if (!imagePath) return undefined
  const cleaned = imagePath.replace(/^\.\//, '').replace(/^\/+/, '')
  if (cleaned.startsWith('data/')) return `/${cleaned}`
  return `/data/${cleaned}`
}

const normalizeWhitespace = (value: string) => value.replace(/\s+/g, ' ').trim()

const parseCollaborationText = (text: string): ParsedCollaboration => {
  const lines = text.split(/\r?\n/)
  const rounds: Round[] = []
  let currentRound: Round = { round: 1, opinions: [] }
  rounds.push(currentRound)

  type Phase = 'opinion' | 'synthesis' | 'review' | 'decision'
  let phase: Phase = 'opinion'
  let currentOpinion: Opinion | null = null
  let currentReview: Review | null = null
  let field: 'explanation' | 'review_reason' | 'review_explanation' | null = null

  const pushOpinion = () => {
    if (currentOpinion) currentRound.opinions.push(currentOpinion)
    currentOpinion = null
    field = null
  }

  const pushReview = () => {
    if (currentReview) {
      currentRound.reviews = currentRound.reviews ?? []
      currentRound.reviews.push(currentReview)
    }
    currentReview = null
    field = null
  }

  const ensureRound = (roundNumber: number) => {
    const existing = rounds.find((r) => r.round === roundNumber)
    if (existing) {
      currentRound = existing
      return
    }
    currentRound = { round: roundNumber, opinions: [] }
    rounds.push(currentRound)
  }

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i] ?? ''
    const line = raw.trim()
    if (!line) continue

    const roundMatch = line.match(/---\s*\[Round\s*(\d+)\]\s*---/i)
    if (roundMatch) {
      pushOpinion()
      pushReview()
      ensureRound(Number(roundMatch[1]))
      phase = 'opinion'
      continue
    }

    if (/Multi-Agent Collaborative Discussion Phase:/i.test(line)) {
      pushOpinion()
      pushReview()
      phase = 'synthesis'
      continue
    }

    if (/This stage encompasses a review from domain agent/i.test(line)) {
      pushOpinion()
      pushReview()
      phase = 'review'
      continue
    }

    if (/This stage encompasses the final decision-making process/i.test(line)) {
      pushOpinion()
      pushReview()
      phase = 'decision'
      continue
    }

    const agentMatch = line.match(/agent ID:\s*(.+?)\s*\((?:role|Role):\s*(.+?)\)/i)
    if (agentMatch) {
      if (phase === 'review') {
        pushReview()
        currentReview = {
          agent_id: normalizeWhitespace(agentMatch[1]),
          specialty: normalizeWhitespace(agentMatch[2]),
          log: { parsed_output: {} },
        }
      } else {
        pushOpinion()
        currentOpinion = {
          agent_id: normalizeWhitespace(agentMatch[1]),
          specialty: normalizeWhitespace(agentMatch[2]),
          log: { parsed_output: {} },
        }
      }
      continue
    }

    if (phase === 'opinion' && currentOpinion) {
      if (line.toLowerCase().startsWith('answer:')) {
        currentOpinion.log.parsed_output.answer = normalizeWhitespace(line.replace(/^answer:/i, ''))
        field = null
        continue
      }
      if (line.toLowerCase().startsWith('explanation:')) {
        currentOpinion.log.parsed_output.explanation = normalizeWhitespace(line.replace(/^explanation:/i, ''))
        field = 'explanation'
        continue
      }
      if (field === 'explanation') {
        currentOpinion.log.parsed_output.explanation = normalizeWhitespace(
          `${currentOpinion.log.parsed_output.explanation ?? ''} ${line}`,
        )
        continue
      }
    }

    if (phase === 'review' && currentReview) {
      if (line.toLowerCase().startsWith('review_result:')) {
        const value = normalizeWhitespace(line.replace(/^review_result:/i, ''))
        currentReview.log.parsed_output.agree = value.toLowerCase() === 'true'
        field = null
        continue
      }
      if (line.toLowerCase().startsWith('review_reason:')) {
        const value = normalizeWhitespace(line.replace(/^review_reason:/i, ''))
        if (value && value.toLowerCase() !== 'n/a') {
          currentReview.log.parsed_output.reason = value
        }
        field = 'review_reason'
        continue
      }
      if (line.toLowerCase().startsWith('review_explanation:')) {
        const value = normalizeWhitespace(line.replace(/^review_explanation:/i, ''))
        if (value && value.toLowerCase() !== 'n/a') {
          currentReview.log.parsed_output.explanation = value
        }
        field = 'review_explanation'
        continue
      }
      if (line.toLowerCase().startsWith('review_answer:')) {
        const value = normalizeWhitespace(line.replace(/^review_answer:/i, ''))
        if (value && value.toLowerCase() !== 'n/a') {
          currentReview.log.parsed_output.answer = value
        }
        field = null
        continue
      }
      if (field === 'review_reason' && currentReview.log.parsed_output.reason) {
        currentReview.log.parsed_output.reason = normalizeWhitespace(`${currentReview.log.parsed_output.reason} ${line}`)
        continue
      }
      if (field === 'review_explanation' && currentReview.log.parsed_output.explanation) {
        currentReview.log.parsed_output.explanation = normalizeWhitespace(
          `${currentReview.log.parsed_output.explanation} ${line}`,
        )
        continue
      }
    }

    if (phase === 'synthesis') {
      if (line.toLowerCase().startsWith('synthesizer answer:')) {
        currentRound.synthesis = currentRound.synthesis ?? { parsed_output: {} }
        currentRound.synthesis.parsed_output.answer = normalizeWhitespace(line.replace(/^synthesizer answer:/i, ''))
        continue
      }
      if (line.toLowerCase().startsWith('synthesizer explanation:')) {
        currentRound.synthesis = currentRound.synthesis ?? { parsed_output: {} }
        currentRound.synthesis.parsed_output.explanation = normalizeWhitespace(
          line.replace(/^synthesizer explanation:/i, ''),
        )
        field = null
        continue
      }
      if (currentRound.synthesis?.parsed_output.explanation) {
        currentRound.synthesis.parsed_output.explanation = normalizeWhitespace(
          `${currentRound.synthesis.parsed_output.explanation} ${line}`,
        )
        continue
      }
    }

    if (phase === 'decision') {
      if (line.toLowerCase().startsWith('decision answer:')) {
        currentRound.decision = currentRound.decision ?? { parsed_output: {} }
        currentRound.decision.parsed_output.answer = normalizeWhitespace(line.replace(/^decision answer:/i, ''))
        continue
      }
      if (line.toLowerCase().startsWith('decision explanation:')) {
        currentRound.decision = currentRound.decision ?? { parsed_output: {} }
        currentRound.decision.parsed_output.explanation = normalizeWhitespace(
          line.replace(/^decision explanation:/i, ''),
        )
        field = null
        continue
      }
      if (currentRound.decision?.parsed_output.explanation) {
        currentRound.decision.parsed_output.explanation = normalizeWhitespace(
          `${currentRound.decision.parsed_output.explanation} ${line}`,
        )
        continue
      }
    }
  }

  pushOpinion()
  pushReview()

  return { raw: text, case_history: { rounds } }
}

export const parseOpenCodingJsonl = (content: string, fileName: string): OpenCodingCase[] => {
  const base = fileName.replace(/_open_coding_human_evl_structured\.jsonl$/i, '')
  const parts = base.split('_')
  const framework = normalizeFramework(parts[0] ?? 'Unknown')
  const dataset = normalizeDataset(parts[1] ?? 'Unknown')
  const modality = dataset.toLowerCase().includes('vqa') || dataset.toLowerCase().includes('slake') || dataset.toLowerCase().includes('pathvqa')
    ? 'vqa'
    : 'text'

  return content
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as JsonlRecord)
    .map((payload, idx) => {
      const questionDescription = String(payload.question_description ?? '').trim()
      const question = parseQuestion(questionDescription)
      const options = parseOptions(questionDescription)
      const answer = parseAnswer(questionDescription)
      const collaborationLog = payload.collaboration_text
        ? parseCollaborationText(payload.collaboration_text)
        : (payload.collaboration_log as ParsedCollaboration | undefined) ?? payload

      const caseId = payload.qid || payload.case_id || payload.caseId || `${slugify(dataset)}-${idx + 1}`
      const imagePath = resolveImagePath(payload.image_path)

      return {
        caseId,
        dataset,
        framework,
        modality,
        question,
        options,
        answer,
        ...(imagePath ? { image: { path: imagePath, alt: `${dataset} image` } } : null),
        collaborationLog,
      }
    })
}
