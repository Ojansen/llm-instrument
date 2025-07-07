import { useRuntimeConfig } from '#imports'

export async function useInference(prompt: string): Promise<string> {
  const base_url = useRuntimeConfig().llmInstruments.baseUrl

  const resp: { output: string } = await $fetch('/inference', {
    baseURL: base_url,
    immediate: false,
    query: {
      query: prompt,
    },
  })

  return resp.output
}
