const ALLOWED_ROOTS = new Set(['audit', 'open-coding'])

const toPathSegments = (value) => {
  if (Array.isArray(value)) return value
  if (typeof value === 'string') return value.split('/').filter(Boolean)
  return []
}

const contentTypeFor = (key) => {
  if (key.endsWith('.json')) return 'application/json; charset=utf-8'
  if (key.endsWith('.jsonl')) return 'application/x-ndjson; charset=utf-8'
  if (key.endsWith('.log')) return 'text/plain; charset=utf-8'
  return 'application/octet-stream'
}

const joinKey = (prefix, relativePath) => {
  const normalizedPrefix = (prefix ?? 'data').replace(/^\/+|\/+$/g, '')
  return normalizedPrefix ? `${normalizedPrefix}/${relativePath}` : relativePath
}

const getObjectKey = (context) => {
  const bucket = context.env.DATA_BUCKET
  if (!bucket) {
    return { response: new Response('Missing R2 binding: DATA_BUCKET', { status: 500 }) }
  }

  const segments = toPathSegments(context.params.path)
  const root = segments[0]
  if (!root || !ALLOWED_ROOTS.has(root) || segments.some((segment) => segment === '..')) {
    return { response: new Response('Not found', { status: 404 }) }
  }

  const relativePath = segments.join('/')
  return { bucket, objectKey: joinKey(context.env.DATA_BUCKET_PREFIX, relativePath) }
}

const responseHeaders = (object, objectKey) => {
  const headers = new Headers()
  object.writeHttpMetadata(headers)
  headers.set('etag', object.httpEtag)
  headers.set('content-type', headers.get('content-type') ?? contentTypeFor(objectKey))
  headers.set(
    'cache-control',
    objectKey.endsWith('/index.json') ? 'no-store' : 'public, max-age=300',
  )
  return headers
}

export async function onRequestGet(context) {
  const result = getObjectKey(context)
  if (result.response) return result.response

  const object = await result.bucket.get(result.objectKey)
  if (!object) {
    return new Response(`R2 object not found: ${result.objectKey}`, { status: 404 })
  }

  return new Response(object.body, { headers: responseHeaders(object, result.objectKey) })
}

export async function onRequestHead(context) {
  const result = getObjectKey(context)
  if (result.response) return result.response

  const object = await result.bucket.head(result.objectKey)
  if (!object) {
    return new Response(null, { status: 404 })
  }

  return new Response(null, { headers: responseHeaders(object, result.objectKey) })
}
