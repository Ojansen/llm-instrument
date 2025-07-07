<template>
  <div>
      <input v-model="query">
      <button @click="hello">Hello</button>
      {{resp}}

  </div>
</template>

<script setup lang="ts">
const resp = ref('');
const query = ref('');

async function hello() {
  const url = `/hello?query=${query.value}`;
  const {data} = await useFetch(url, {
    baseURL: 'http://localhost',
  });
  resp.value = data.value.message;
}
</script>
