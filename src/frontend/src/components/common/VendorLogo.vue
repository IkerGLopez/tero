<script setup lang="ts">
import { computed } from 'vue';
import { LlmModelVendor } from '@/services/api';
import openAiIsotipo from '@/assets/images/openai-iso.svg?raw';
import googleIsotipo from '@/assets/images/gemini-iso.svg?raw';
import anthropicIsotipo from '@/assets/images/anthropic-iso.svg?raw';
import qwenIsotipo from '@/assets/images/qwen-iso.svg?raw';

const vendorLogos: Record<LlmModelVendor, string> = {
  [LlmModelVendor.OPENAI]: openAiIsotipo,
  [LlmModelVendor.ANTHROPIC]: anthropicIsotipo,
  [LlmModelVendor.GOOGLE]: googleIsotipo,
  [LlmModelVendor.QWEN]: qwenIsotipo
};

const props = defineProps<{
  vendor: LlmModelVendor;
}>();

const icon = computed(() => vendorLogos[props.vendor]);
</script>

<template>
  <span v-html="icon" class="flex items-center" />
</template>
