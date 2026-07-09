<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconAlertCircle, IconBox, IconInfoCircleFilled, IconTrashX } from '@tabler/icons-vue'
import { truncateFileName } from '../../../../common/src/utils/file'

const props = defineProps<{
  errorKey?: string | null
  fileName?: string
}>()

defineEmits<{
  (e: 'close'): void
  (e: 'remove-file'): void
}>()

const { t } = useI18n()
const errors = computed(() => (props.errorKey ? [t(props.errorKey)] : []))
</script>

<template>
  <div class="flex flex-col gap-4 pt-6">
    <div class="flex flex-col gap-2">
      <IconAlertCircle size="80" stroke-width="1" class="text-error-alt" />
      <span class="bold-span pt-4 text-lg">{{ t('importErrorTitle') }}</span>
      <span>{{ t('importErrorDescription') }}</span>
    </div>

    <div class="mb-4 flex w-full flex-row gap-2 rounded-lg bg-surface-muted p-4 px-2 text-sm">
      <IconInfoCircleFilled size="24" class="text-content" />
      <div class="flex flex-col">
        <span v-for="(errorTitle, index) in errors" :key="index" class="bold-span">{{ errorTitle }}</span>
      </div>
    </div>

    <div v-if="fileName" class="flex flex-row items-center justify-between rounded-lg border border-error-alt p-2">
      <div class="flex flex-row items-center gap-2">
        <IconBox class="text-content-muted" />
        {{ truncateFileName(fileName) }}
      </div>
      <SimpleIcon interactive @click="$emit('remove-file')" :icon="IconTrashX" class="text-content-muted" />
    </div>

    <div class="mt-6 flex flex-row justify-end gap-4">
      <SimpleButton @click="$emit('close')" shape="square">{{ t('close') }}</SimpleButton>
      <SimpleButton v-if="fileName" @click="$emit('remove-file')" variant="primary" shape="square">{{ t('tryAgain') }}</SimpleButton>
    </div>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "invalidFileFormat": "Invalid file format",
    "missingRequiredConfiguration": "Missing required configuration",
    "unsupportedFileStructure": "Unsupported file structure",
    "importErrorTitle": "We couldn’t import this file",
    "importErrorDescription": "The file couldn’t be imported. Check that it’s a valid ZIP and try again.",
    "close": "Close",
    "tryAgain": "Try again"
  },
  "es": {
    "invalidFileFormat": "Formato de archivo inválido",
    "missingRequiredConfiguration": "Falta configuración requerida",
    "unsupportedFileStructure": "Estructura de archivo no soportada",
    "importErrorTitle": "No pudimos importar este archivo",
    "importErrorDescription": "No se pudo importar el archivo. Verifica que sea un ZIP válido e inténtalo de nuevo.",
    "close": "Cerrar",
    "tryAgain": "Intentar de nuevo"
  }
}
</i18n>
