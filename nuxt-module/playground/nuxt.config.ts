export default defineNuxtConfig({
  modules: ['../src/module'],
  devtools: { enabled: true },

  runtimeConfig: {
    llmInstruments: {
      baseUrl: 'http://localhost',
    },
  },
})
