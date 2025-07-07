import { useRuntimeConfig } from '#imports'

export async function useSimilarity(prompt: string, reference: string): Promise<{ score: string, passing: string }> {
  const base_url = useRuntimeConfig().llmInstruments.baseUrl

  return await $fetch('/similarity', {
    baseURL: base_url,
    immediate: false,
    query: {
      prompt: prompt,
      reference: reference,
    },
  })
}
