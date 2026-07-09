<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps<{
  id: string;
  modelValue: string[] | undefined;
  label: string;
  optionValues: string[];
  optionLabels: (propName: string, value: string) => string;
  viewMode?: boolean;
  description?: string | null;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: string[]): void
}>();

const { t } = useI18n()
const safeModelValue = computed(() => props.modelValue || [])

const isSelected = (val: string) => safeModelValue.value.includes(val)

const toggleOption = (val: string) => {
  if (props.viewMode) return
  if (isSelected(val)) {
    emit('update:modelValue', safeModelValue.value.filter(option => option !== val))
  } else {
    emit('update:modelValue', [...safeModelValue.value, val])
  }
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <label class="font-semibold text-sm">{{ label }}</label>
    <div class="flex flex-row flex-wrap gap-3">
      <SimpleButton
        v-for="option in optionValues"
        :key="option"
        :variant="isSelected(option) ? 'primary' : undefined"
        shape="square"
        size="small"
        class="p-2! py-1!"
        @click="toggleOption(option)"
      >
        {{ optionLabels(id, option) }}
      </SimpleButton>
    </div>
    <p
      class="text-sm text-content-muted"
      v-html="description || t('selectAtLeastOne')">
    </p>
  </div>
</template>

<i18n lang="json">
  {
    "en": {
      "selectAtLeastOne": "Select at least one option."
    },
    "es": {
      "selectAtLeastOne": "Selecciona al menos una opcion."
    }
  }
</i18n>
