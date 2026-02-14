export type TaxonomyKey = keyof typeof FAILURE_MODE_DEFINITION_MAPPING

export type TaxonomyItem = {
  key: TaxonomyKey
  title: string
  short: string
  definition: string
  human_eval_instruction: string
}

import { FAILURE_MODE_DEFINITION_MAPPING } from './failureModes'

export const TAXONOMY: TaxonomyItem[] = Object.entries(FAILURE_MODE_DEFINITION_MAPPING).map(([key, value]) => ({
  key: key as TaxonomyKey,
  title: value.name,
  short: value.definition,
  definition: value.definition,
  human_eval_instruction: value.human_eval_instruction,
}))

