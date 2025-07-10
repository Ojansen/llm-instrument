import { useMetrics } from '../../../src/runtime/composables/useMetrics'

export default defineEventHandler(async () => {
  const { faithfulness, correctness, similarity } = useMetrics()
  const prompt = 'the color of the sky is ...'
  const reference = 'blue'

  const sim = await similarity(prompt, reference)
  const corr = await correctness(prompt, reference)
  const faith = await faithfulness(prompt)

  return {
    sim, corr, faith,
  }
})
