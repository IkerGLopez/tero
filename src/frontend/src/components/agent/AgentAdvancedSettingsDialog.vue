<script setup lang="ts">
import { Dialog } from 'primevue'
import { useI18n } from 'vue-i18n'
import { IconSettings, IconX } from '@tabler/icons-vue'
import SavingIndicator from '@/components/common/SavingIndicator.vue'
const { t } = useI18n()
const props = defineProps<{
  recursionLimit: number
  saving: boolean
}>()

const emit = defineEmits<{
  (e: 'update', value: number): void
}>()

const visible = defineModel<boolean>("visible")

const RECURSION_LIMIT_OPTIONS = [20, 40, 50, 80, 100, 130]

</script>

<template>
  <Dialog v-model:visible="visible" :modal="true" :draggable="false" :resizable="false" :closable="false" class="basic-dialog max-w-2xl">
    <FlexCard>
      <template #header>
        <div class="flex flex-row items-center justify-between">
          <div class="flex gap-2 items-center">
            <IconSettings />
            <span> | {{ t('advancedSettingsTitle') }}</span>
            <SavingIndicator v-if="props.saving" />
          </div>
          <SimpleButton @click="visible = false">
            <IconX />
          </SimpleButton>
        </div>
      </template>
      <div class="mb-4">
        <div class="form-field flex flex-col gap-2">
            <label for="recursionLimit">{{ t('recursionLimit') }}</label>
            <p class="text-content-muted text-sm tool-message mb-2 whitespace-pre-line">{{ t('recursionLimitHelp') }}</p>
            <SelectButton
              id="recursionLimit"
              :model-value="props.recursionLimit"
              :options="RECURSION_LIMIT_OPTIONS.map(v => ({ label: String(v), value: v }))"
              option-label="label"
              option-value="value"
              :allow-empty="false"
              @update:model-value="emit('update', $event)"
              class="w-fit"
            />
        </div>
      </div>
    </FlexCard>
  </Dialog>
</template>


<i18n lang="json">
{
  "en": {
    "advancedSettingsTitle": "Advanced settings",
    "recursionLimit": "Thought process steps limit",
    "recursionLimitHelp": "Higher values increase budget usage and response time. Before increasing, optimize your agent to use fewer steps."
  },
  "es": {
    "advancedSettingsTitle": "Configuración avanzada",
    "recursionLimit": "Límite de pasos del proceso de pensamiento",
    "recursionLimitHelp": "Valores más altos aumentan el uso de presupuesto y el tiempo de respuesta. Antes de aumentar, optimiza tu agente para usar menos pasos."
  }
}
</i18n>
