<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { ApiService, LlmModel, LlmTemperature, ReasoningEffort } from '@/services/api'
import { Evaluator } from '@/services/api'
import LlmModelSettings from './LlmModelSettings.vue'
import { useErrorHandler } from '@/composables/useErrorHandler'

const props = defineProps<{
  showModal: boolean
  agentId: number
  testCaseId?: number
  testCaseName?: string
}>()

const emit = defineEmits<{
  (e: 'update:showModal', value: boolean): void
  (e: 'save', evaluator: Evaluator): void
}>()

const { t } = useI18n()
const { handleError } = useErrorHandler()

const api = new ApiService()
const isLoading = ref(false)
const modelId = ref<string>('')
const temperature = ref<LlmTemperature>(LlmTemperature.NEUTRAL)
const reasoningEffort = ref<ReasoningEffort>(ReasoningEffort.MEDIUM)
const prompt = ref<string>('')
const models = ref<LlmModel[]>([])

watch(() => props.showModal, async (isOpen) => {
  if (isOpen) {
    await loadEvaluator()
  }
})

const loadEvaluator = async () => {
  isLoading.value = true
  try {
    models.value = await api.findModels()
    const config = props.testCaseId
      ? await api.findTestCaseEvaluator(props.agentId, props.testCaseId)
      : await api.findAgentEvaluator(props.agentId)
    modelId.value = config.modelId || ''
    temperature.value = config.temperature || LlmTemperature.NEUTRAL
    reasoningEffort.value = config.reasoningEffort || ReasoningEffort.MEDIUM
    prompt.value = config.prompt || ''
  } catch (error) {
    handleError(error)
  } finally {
    isLoading.value = false
  }
}

const evaluatorModel = computed(() => models.value.find(m => m.id === modelId.value))

const onClose = () => {
  emit('update:showModal', false)
}

const onSave = () => {
  emit('save', new Evaluator(
    modelId.value,
    temperature.value,
    reasoningEffort.value,
    prompt.value
  ))
  emit('update:showModal', false)
}
</script>

<template>
  <Dialog
    :visible="showModal"
    @update:visible="emit('update:showModal', $event)"
    :modal="true"
    :draggable="false"
    :resizable="false"
    :closable="false"
    class="w-200"
  >
    <template #header>
      <div class="flex flex-col gap-2 w-full border-b pb-4">
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-2">
            <IconClipboardCheck />
            <h3>{{ testCaseId && testCaseName ? t('evaluatorTitleForTestCase', { testCaseName }) : t('evaluatorTitle') }}</h3>
          </div>
          <SimpleButton @click="onClose"><IconX /></SimpleButton>
        </div>
        <i18n-t v-if="testCaseId" keypath="testCaseEvaluatorNote" tag="p" class="text-sm text-content-muted">
          <template #testCaseName>
            <b class="text-content">{{ testCaseName }}</b>
          </template>
        </i18n-t>
      </div>
    </template>
    <div v-if="isLoading" class="flex flex-col gap-4 animate-pulse">
      <!-- Model and settings row -->
      <div class="flex flex-row gap-3">
        <div class="form-field w-80">
          <div class="h-4 bg-auxiliar-gray rounded w-16 mb-2"></div>
          <div class="h-10 bg-auxiliar-gray rounded w-full"></div>
        </div>
        <div class="flex-1"></div>
        <div class="form-field">
          <div class="h-4 bg-auxiliar-gray rounded w-24 mb-2"></div>
          <div class="h-10 bg-auxiliar-gray rounded w-full"></div>
        </div>
      </div>
      <!-- Instructions field -->
      <div class="form-field">
        <div class="h-4 bg-auxiliar-gray rounded w-28 mb-2"></div>
        <div class="h-40 bg-auxiliar-gray rounded w-full"></div>
      </div>
    </div>
    <div v-else class="flex flex-col gap-4">
      <div class="flex flex-row gap-3">
        <div class="form-field gap-2">
          <label>{{ t('modelLabel') }}</label>
          <Select
            v-model="modelId"
            :options="models"
            option-label="name"
            option-value="id"
            class="w-80"
          >
            <template #option="slotProps">
              <div class="flex flex-col gap-2">
                <div class="flex items-center gap-2">
                  <span>{{ slotProps.option.name }}</span>
                </div>
              </div>
            </template>
            <template #dropdownicon>
              <IconCaretRightFilled />
            </template>
          </Select>
        </div>
        <div class="flex-1"></div>
        <LlmModelSettings
          :model-type="evaluatorModel?.modelType"
          v-model:temperature="temperature"
          v-model:reasoning-effort="reasoningEffort"
        />
      </div>
      <div class="form-field relative">
        <div class="flex flex-col gap-1">
          <label>{{ t('instructionsLabel') }}</label>
        </div>
        <InteractiveInput
          v-model="prompt"
          :rows="10"
          :placeholder="t('instructionsPlaceholder')"
        />
      </div>
    </div>
    <template #footer>
      <div class="flex gap-4">
        <SimpleButton @click="onClose" shape="square">
          {{ t('cancel') }}
        </SimpleButton>
        <SimpleButton @click="onSave" shape="square" variant="primary">
          {{ t('confirm') }}
        </SimpleButton>
      </div>
    </template>
  </Dialog>
</template>

<i18n lang="json">
{
  "en": {
    "evaluator": "evaluator",
    "evaluatorTitle": "Agent evaluator",
    "evaluatorTitleForTestCase": "Test evaluator",
    "testCaseEvaluatorNote": "This evaluator configuration will only be used for the test {testCaseName}.",
    "modelLabel": "Model",
    "instructionsLabel": "Instructions",
    "instructionsPlaceholder": "Write the instructions for this evaluator",
    "cancel": "Cancel",
    "confirm": "Confirm"

  },
  "es": {
    "evaluator": "evaluador",
    "evaluatorTitle": "Evaluador de agente",
    "evaluatorTitleForTestCase": "Evaluador de test",
    "testCaseEvaluatorNote": "Esta configuración del evaluador solo se usará para el test {testCaseName}.",
    "modelLabel": "Modelo",
    "instructionsLabel": "Instrucciones",
    "instructionsPlaceholder": "Escribe las instrucciones para este evaluador",
    "cancel": "Cancelar",
    "confirm": "Confirmar"

  }
}
</i18n>
