const escapeHtml = (input: string) =>
  input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

// Minimal, safe markdown rendering for our logs:
// - Escapes all HTML
// - Supports: headings, bold, italics, inline code, fenced code blocks, unordered lists, paragraphs, line breaks
export const renderMarkdownSafe = (markdown: string): string => {
  const input = markdown.replace(/\r\n/g, '\n')
  const lines = input.split('\n')

  const out: string[] = []
  let inCode = false
  let codeLang = ''
  let codeLines: string[] = []
  let inList = false

  const flushList = () => {
    if (!inList) return
    out.push('</ul>')
    inList = false
  }

  const flushCode = () => {
    if (!inCode) return
    const code = escapeHtml(codeLines.join('\n'))
    const langClass = codeLang ? ` class=\"language-${escapeHtml(codeLang)}\"` : ''
    out.push(`<pre class=\"whitespace-pre-wrap\"><code${langClass}>${code}</code></pre>`)
    inCode = false
    codeLang = ''
    codeLines = []
  }

  const renderInline = (text: string) => {
    let s = escapeHtml(text)
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>')
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>')
    return s
  }

  for (const raw of lines) {
    const line = raw

    const fence = line.match(/^```(\w+)?\s*$/)
    if (fence) {
      if (inCode) {
        flushCode()
      } else {
        flushList()
        inCode = true
        codeLang = fence[1] ?? ''
      }
      continue
    }

    if (inCode) {
      codeLines.push(line)
      continue
    }

    if (!line.trim()) {
      flushList()
      continue
    }

    const h = line.match(/^(#{1,3})\s+(.*)$/)
    if (h?.[1]) {
      flushList()
      const level = h[1].length
      out.push(`<h${level} class=\"mt-2 font-semibold\">${renderInline(h[2] ?? '')}</h${level}>`)
      continue
    }

    const li = line.match(/^\s*[-*]\s+(.*)$/)
    if (li) {
      if (!inList) {
        out.push('<ul class=\"list-disc pl-5\">')
        inList = true
      }
      out.push(`<li>${renderInline(li[1] ?? '')}</li>`)
      continue
    }

    flushList()
    out.push(`<p>${renderInline(line)}</p>`)
  }

  flushList()
  flushCode()
  return out.join('\n')
}
