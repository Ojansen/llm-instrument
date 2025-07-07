<template>
  <div>
    <button @click="hello">Hello</button>
  </div>
</template>

<script setup lang="ts">


async function hello() {
  const {data} = await useFetch('/hello?name=test', {
    baseURL: 'http://localhost',
  });
}
</script>
