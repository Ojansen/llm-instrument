<template>
  <div>
    Nuxt module playground!
    <button
      type="button"
      @click="get"
    >
      Submit
    </button>
  </div>
</template>

<script setup>
function get() {
  const {} = $fetch('/api/generate')
}
</script>
