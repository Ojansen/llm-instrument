import { useInference } from '../../../src/runtime/composables/useInference'
import { useSimilarity } from '../../../src/runtime/composables/useSimilarity'

export default defineEventHandler((event) => {
  // return useInference('the sky is ...')
  return useSimilarity('the sky is ...', 'blue')
})
