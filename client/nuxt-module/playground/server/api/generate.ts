import { useInference } from '../../../src/runtime/composables/useInference'
import { useSimilarity } from '../../../src/runtime/composables/useSimilarity'
import { useCorrectness } from '../../../src/runtime/composables/useCorrectness';

export default defineEventHandler(() => {
  // return useInference('the sky is ...')
  // return useSimilarity('the sky is ...', 'blue')
  return useCorrectness('the sky is ...', 'blue')
})
