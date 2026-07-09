<script setup lang="ts">
import { IconLock } from '@tabler/icons-vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()


</script>

<template>
  <div class="flex flex-col gap-3">
    <div class="flex p-4 rounded-2xl bg-surface-muted gap-6 items-center">
      <div class="bg-content-muted rounded-2xl p-2">
        <IconLock class="text-white" />
      </div>
      <span class="font-semibold text-content-muted">{{ t('protected') }}</span>
      <span class="text-content-muted" v-html="t('protectedDescription')"></span>
    </div>
    <span class="text-content-muted text-sm">{{ t('protectedUsage') }}</span>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "protected": "Protected configuration",
    "protectedDescription": "This agent has its configuration protected by its author. {'<'}span class='font-semibold'>It is not possible to view, clone or modify its configuration.{'<'}/span>",
    "protectedUsage": "You can use it normally, but you cannot change its behavior."
  },
  "es": {
    "protected": "Configuración protegida",
    "protectedDescription": "Este agente tiene su configuración protegida por su autor. {'<'}span class='font-semibold'>No es posible ver, clonar ni modificar su configuración.{'<'}/span>",
    "protectedUsage": "Podés usarlo normalmente, pero no cambiar su comportamiento."
  }
}
</i18n>
