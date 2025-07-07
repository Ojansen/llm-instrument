// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-05-15',
  devtools: { enabled: true },

  modules: [
    '@nuxt/scripts',
    '@nuxt/ui',
    '@nuxt/test-utils',
    '@nuxt/image',
    '@nuxt/icon',
    '@nuxt/fonts'
  ]
})