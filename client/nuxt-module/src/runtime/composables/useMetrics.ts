import { useRuntimeConfig } from '#imports'

const MetricTypes = {
  similarity: 'similarity',
  correctness: 'correctness',
  faithfulness: 'faithfulness',
} as const

type MetricKey = typeof MetricTypes[keyof typeof MetricTypes]

export function useMetrics() {
  const base_url = useRuntimeConfig().llmInstruments.baseUrl

  function _get(url: MetricKey, query: Record<string, string>) {
    return $fetch(`/metrics/${url}`, {
      baseURL: base_url,
      immediate: false,
      query: query,
    })
  }

  async function _common(metricType: MetricKey, prompt: string, reference = '') {
    return await _get(metricType, {
      prompt: prompt,
      reference: reference,
    })
  }

  return {
    faithfulness: (prompt: string) => _common(MetricTypes.faithfulness, prompt),
    correctness: (prompt: string, reference: string) => _common(MetricTypes.correctness, prompt, reference),
    similarity: (prompt: string, reference: string) => _common(MetricTypes.similarity, prompt, reference),
  }
}
