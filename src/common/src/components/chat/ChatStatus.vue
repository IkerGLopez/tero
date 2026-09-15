<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconChevronDown, IconPointFilled } from '@tabler/icons-vue'
import type { StatusUpdate } from './ChatMessage.vue'
import { truncatePreview } from '../../utils/textPreview'

const { t } = useI18n()

const props = defineProps<{
  statusUpdates: StatusUpdate[]
  isComplete: boolean
}>()

const isExpanded = ref(false)

const showStatus = computed(() => {
  return props.statusUpdates.length > 0
})

const currentStatusText = computed(() => {
  if (!props.statusUpdates?.length) return ''
  return formatStatusAction(props.statusUpdates[props.statusUpdates.length - 1])
})

const thoughtTimeInSeconds = computed(() => {
  if (props.statusUpdates.length === 0) return 0
  const firstUpdate = props.statusUpdates[0]
  const lastUpdate = props.statusUpdates[props.statusUpdates.length - 1]
  if (!firstUpdate.timestamp || !lastUpdate.timestamp) return 0
  const diffMs = lastUpdate.timestamp.getTime() - firstUpdate.timestamp.getTime()
  return Math.round(diffMs / 1000)
})

const toggleExpanded = () => {
  isExpanded.value = !isExpanded.value
}

const formatStatusAction = (status: StatusUpdate): string => {
  switch (status.action) {
    case 'statusProcessing':
      return t('statusProcessing')
    case 'preModelHook':
      return t('statusPreModel')
    case 'planning':
      return t('planning', { tool: status.toolName })
    case 'executingTool':
      return t('statusExecutingTool', { tool: status.toolName }) 
      + (status.step ? " - " + t(status.step) : '') 
      + (status.args? t('withParams', { params: status.args })  : '')
    case 'executedTool':
      return t('statusExecutedTool', { tool: status.toolName })
    case 'toolError':
      return t('toolError', { tool: status.toolName})
    default:
      return status.action
  }
}

watch(() => props.isComplete, (newIsComplete) => {
  if (newIsComplete) {
    isExpanded.value = false
  }
})
</script>

<template>
  <div v-if="showStatus" class="status-container mb-2">
    <div class="overflow-hidden border-b">
      <button @click="toggleExpanded" class="w-full flex items-center justify-between py-2">
        <div v-if="!isComplete" class="flex items-center gap-2 w-full text-left">
          <div class="w-3 h-3 p-1 border-2 border-t-transparent rounded-full animate-spin"></div>
          <span class="text-sm text-content-muted" v-html="currentStatusText"></span>
        </div>
        <div v-else class="flex items-center gap-2">
          <span class="text-sm font-medium text-content-muted">
            {{ thoughtTimeInSeconds > 0 ? t('endMessage', { time: thoughtTimeInSeconds }) : t('thoughtProcessMessage') }}
          </span>
        </div>
        <IconChevronDown :class="['w-4 h-4 transition-transform', { 'rotate-180': isExpanded }]" />
      </button>
      <div v-if="isExpanded" class="mb-2 overflow-y-auto" :class="{ 'max-h-60': isComplete }">
        <div v-for="(status, index) in statusUpdates" :key="index" class="flex items-stretch gap-1 text-sm animate-fade-in opacity-0" :style="{ animationDelay: `${index * 0.1}s` }" >
          <div class="flex flex-col items-center">
            <IconPointFilled class="text-content-muted" size="16"/>
            <span v-if="index !== statusUpdates.length - 1" class="w-[2px] flex-1 bg-auxiliar-gray"></span>
          </div>
          <div class="w-full pb-3">
            <div class="flex items-center gap-2 left-[-2px]">
              <span v-html="formatStatusAction(status)"></span>
            </div>
            <div v-if="status.description && status.description.trim() !== ''">
              <span>{{ t('description') }}: </span> <span class="italic">{{ status.description }}</span>
            </div>
            <div v-if="status.result && (typeof status.result === 'string' ? status.result.trim() !== '' : false)">
              {{ t('result') }} {{ status.result }}
            </div>
            <div v-if="status.result && Array.isArray(status.result) && status.result.length > 0">
              <div v-for="doc in status.result" class="flex items-start gap-1 text-sm">-<b>{{ truncatePreview(doc) }}</b></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "statusProcessing": "Processing",
    "statusPreModel": "Thinking",
    "planning": "Planning to run tools",
    "statusExecutingTool": "Executing {'<'}b>{tool}{'<'}/b>",
    "statusExecutedTool": "Tool {'<'}b>{tool}{'<'}/b> execution finished",
    "documentsRetrieved": "{count} chunks retrieved",
    "toolError": "Error in tool {'<'}b>{tool}{'<'}/b>",
    "endMessage": "Thought for {time} seconds",
    "result": "Result:",
    "results": "Results {count}:",
    "description": "Description",
    "retrieving": "Retrieving chunks",
    "retrieved": "Chunks retrieved",
    "analyzing": "Analyzing chunks",
    "analyzed": "Chunks analyzed",
    "groundingResponse": "Validating response",
    "groundedResponse": "Response validated",
    "withParams": " with params {'<'}b>{params}{'<'}/b>",
    "thoughtProcessMessage": "Thought process"
  },
  "es": {
    "statusProcessing": "Procesando",
    "statusPreModel": "Pensando",
    "planning": "Planificando ejecutar herramientas",
    "statusExecutingTool": "Ejecutando {'<'}b>{tool}{'<'}/b>",
    "statusExecutedTool": "Ejecución de herramienta {'<'}b>{tool}{'<'}/b> finalizada",
    "documentsRetrieved": "{count} secciones recuperados",
    "toolError": "Error en la herramienta {'<'}b>{tool}{'<'}/b>",
    "endMessage": "Pensó durante {time} segundos",
    "result": "Resultado:",
    "results": "Resultados {count}:",
    "retrieving": "Recuperando secciones",
    "retrieved": "Secciones recuperadas",    
    "analyzing": "Analizando secciones",
    "analyzed": "Secciones analizadas",
    "groundingResponse": "Validando respuesta",
    "groundedResponse": "Respuesta validada",
    "description": "Descripción",
    "withParams": " con los siguientes parametros {'<'}b>{params}{'<'}/b>",
    "thoughtProcessMessage": "Proceso de pensamiento"
  }
}
</i18n>
