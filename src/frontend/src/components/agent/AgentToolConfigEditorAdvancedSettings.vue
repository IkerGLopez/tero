<script lang="ts" setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconCaretDownFilled, IconCaretUpFilled } from '@tabler/icons-vue'
import SimpleIcon from '@tero/common/components/common/SimpleIcon.vue'

withDefaults(
  defineProps<{
    fieldId: string
    label: string
    viewMode?: boolean
  }>(),
  { viewMode: false },
)

const headers = defineModel<Array<Record<string, string>>>({ default: () => [] })

const { t } = useI18n()
const collapsed = ref(headers.value.length === 0)

const addEntry = () => {
  headers.value.push({ name: '', value: '' })
}

const removeEntry = (index: number) => {
  headers.value.splice(index, 1)
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <Panel class="border-none!" toggleable :collapsed="collapsed" @update:collapsed="collapsed = $event">
      <template #header>
        <h4 class="text-sm! font-semibold">{{ t('advancedSettings') }}</h4>
      </template>
      <template #toggleicon>
        <div class="p-1 bg-surface-muted">
          <SimpleIcon interactive :icon="collapsed ? IconCaretDownFilled : IconCaretUpFilled" class="text-content-muted" />
        </div>
      </template>
      <div class="flex flex-col gap-3">
        <div class="flex flex-col gap-1">
          <label :for="fieldId" class="text-sm font-semibold">{{ label }}</label>
          <p class="text-sm text-content-muted">{{ t('customHeadersDescription') }}</p>
        </div>
        <div v-for="(entry, index) in headers" :key="index" class="flex gap-4 items-center">
          <InteractiveInput v-model="entry.name" :placeholder="t('customHeaderName')" :disabled="viewMode" class="flex-1" />
          <InteractiveInput v-model="entry.value" :placeholder="t('customHeaderValue')" :disabled="viewMode" class="flex-1" />
          <SimpleButton class="text-content-muted" :disabled="viewMode" @click="removeEntry(index)">
            <IconTrashX />
          </SimpleButton>
        </div>
        <SimpleButton size="small" shape="square" class="self-start px-2" :disabled="viewMode" @click="addEntry">
          <IconPlus />
          {{ t('customHeaderAdd') }}
        </SimpleButton>
      </div>
    </Panel>
  </div>
</template>

<style scoped>
@import '@/assets/styles.css';

:deep(.p-panel-header),
:deep(.p-panel-content) {
  @apply !px-0.5;
}

:deep(.p-panel-header) {
  @apply border-b mb-6 pt-0!;
}
</style>

<i18n lang="json">
  {
    "en": {
      "advancedSettings": "Advanced settings (optional)",
      "customHeadersDescription": "These headers will be sent when connecting to this server",
      "customHeaderName": "Header name",
      "customHeaderValue": "Value",
      "customHeaderAdd": "Add header"
    },
    "es": {
      "advancedSettings": "Configuración avanzada (opcional)",
      "customHeadersDescription": "Estos headers serán enviados cuando se conecte a este servidor",
      "customHeaderName": "Nombre del header",
      "customHeaderValue": "Valor",
      "customHeaderAdd": "Agregar header"
    }
  }
</i18n>
