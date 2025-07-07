import { useInference } from '../../../src/runtime/composables/useInference'

export default defineEventHandler((event) => {
  return useInference('the sky is ...')
})
