export const fnv1a32 = (input: string): number => {
  let hash = 0x811c9dc5
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}

export const stablePickByWeight = <T>(
  items: readonly T[],
  seed: string,
  score: (item: T) => string,
): T[] => {
  const sorted = [...items].sort((a, b) => score(a).localeCompare(score(b)))
  return sorted.sort((a, b) => fnv1a32(seed + score(a)) - fnv1a32(seed + score(b)))
}

