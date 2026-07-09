<script lang="ts" setup>
import { ref } from 'vue'
import { IconCopy, IconCheck } from '@tabler/icons-vue'

defineProps<{ text: string }>()
const emit = defineEmits<{
  (e: 'error', error: unknown): void
}>()
const copied = ref(false)

const copy = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    setTimeout(() => copied.value = false, 2000)
  } catch (error) {
    emit('error', error)
  }
}
</script>

<template>
  <button @click="copy(text)">
    <IconCopy v-if="!copied" size="18" class="text-content-muted hover:text-abstracta cursor-pointer" />
    <IconCheck v-else size="18" class="text-primary" />
  </button>
</template>
