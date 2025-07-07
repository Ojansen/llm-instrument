import { defineNuxtModule, addPlugin, createResolver, addImportsDir } from '@nuxt/kit'
import defu from 'defu'

// Module options TypeScript interface definition
export interface ModuleOptions {
  baseUrl?: string
}

export default defineNuxtModule<ModuleOptions>({
  meta: {
    name: 'llm-instruments',
    configKey: 'llmInstruments',
  },
  // Default configuration options of the Nuxt module
  defaults: {},
  setup(_options, _nuxt) {
    const resolver = createResolver(import.meta.url)

    _nuxt.options.runtimeConfig.llmInstruments = defu(_nuxt.options.runtimeConfig.llmInstruments, {
      baseUrl: _options.baseUrl,
    })

    // Do not add the extension since the `.ts` will be transpiled to `.mjs` after `npm run prepack`
    addPlugin({
      src: resolver.resolve('./runtime/plugin'),
      mode: 'server',
    })

    addImportsDir(resolver.resolve('runtime/composables'))
  },
})
