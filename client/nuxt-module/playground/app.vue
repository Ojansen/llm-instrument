<template>
  <div>
    <h1>LLM Instruments</h1>
    <button
      type="button"
      @click="get"
    >
      Submit
    </button>
    <pre>
      {{ response }}
    </pre>
  </div>
</template>

<script setup>
const response = ref('')
async function get() {
  response.value = await $fetch('/api/generate')
}
</script>
