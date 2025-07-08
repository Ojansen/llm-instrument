import { useRuntimeConfig } from '#imports'

export async function useCorrectness(prompt: string, reference: string): Promise<{ score: string, passing: string }> {
  const base_url = useRuntimeConfig().llmInstruments.baseUrl

  return await $fetch('/correctness', {
    baseURL: base_url,
    immediate: false,
    query: {
      prompt: prompt,
      reference: reference,
    },
  })
}
