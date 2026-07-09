<script lang="ts">
import { reactive } from 'vue'
import { type Icon } from '@tabler/icons-vue'
import type { JSONSchema7, JSONSchema7Definition } from 'json-schema'
import { useErrorHandler } from '@/composables/useErrorHandler'
import Ajv, { type ErrorObject } from 'ajv'
import addFormats from 'ajv-formats'
import { handleToolAuthRequestsIn } from '@/services/toolAuth';
import { AuthenticationError } from '@tero/common/utils/toolAuth.js';
import { AgentToolConfig, AgentTool } from '@/services/api'

export class EditingToolConfig {
  agentId: number
  tool: AgentTool
  name: string
  icon: Icon
  toolId: string
  config?: Record<string, unknown>

  constructor(agentId: number, tool: AgentTool, name: string, icon: Icon, toolConfig?: AgentToolConfig) {
    this.agentId = agentId
    this.tool = tool
    this.toolId = toolConfig?.toolId ?? tool.id
    this.config = toolConfig?.config
    this.name = name
    this.icon = icon
  }
}
</script>

<script setup lang="ts">
import { ref, computed, onMounted, watchEffect } from 'vue'
import { useI18n } from 'vue-i18n'
import { ApiService, findManifest, HttpError } from '@/services/api'

const props = defineProps<{
  toolConfig: EditingToolConfig
  viewMode?: boolean
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'update', toolConfig: AgentToolConfig): void
}>()

const { t } = useI18n()
const api = new ApiService()
const { handleError } = useErrorHandler()
const validationErrors = ref<string | null>(null)
const saving = ref(false)
const contactEmail = ref<string>('')
const savedConfig = ref<Record<string, unknown> | undefined>()
const allFilesRemoved = ref(false)
const mutableConfig = reactive({ ...props.toolConfig.config || {} })
let partialSave = false

onMounted(async () => {
  try {
    const manifest = await findManifest()
    contactEmail.value = manifest.contactEmail
    savedConfig.value = props.toolConfig.config
    initializeToolConfig()
  } catch (error) {
    handleError(error)
  }
})

const initializeToolConfig = () => {
  Object.entries(toolProperties.value).forEach(([propName, propSchema]) => {
    if (isBooleanProperty(propSchema) && mutableConfig[propName] === undefined) {
      mutableConfig[propName] = false
    }
    if (isStringEnumProperty(propSchema) && mutableConfig[propName] === undefined) {
      const defaultEnumValue = js7(propSchema)!.default
      if (defaultEnumValue) {
        mutableConfig[propName] = defaultEnumValue
      }
    }
  })
}

const toolProperties = computed(() => {
  return props.toolConfig.tool.configSchema.properties!
})

const isBooleanProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'boolean'
}

function js7(val: unknown): JSONSchema7 | undefined {
  return val as JSONSchema7
}

const areDependenciesMet = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  if (toolPropSchema.dependencies === undefined) {
    return true
  }
  return Object.entries(toolPropSchema.dependencies).every(([dependencyKey, dependencyValue]) => {
    const dependencyPropValue = mutableConfig[dependencyKey]
    if (dependencyPropValue === undefined) {
      return false
    }
    const dependencySchema = js7(dependencyValue)
    if (dependencySchema?.const !== undefined) {
      return dependencyPropValue === dependencySchema.const
    }
    return true
  })
}

watchEffect(() => {
  Object.entries(toolProperties.value).forEach(([propName, propSchema]) => {
    if (!areDependenciesMet(propSchema) && mutableConfig[propName] !== undefined) {
      delete (mutableConfig as Record<string, unknown>)[propName]
    }
  })
})

const translateToolPropertyName = (toolId: string, propName: string): string => {
  return t(buildToolPropertyTranslationKey(toolId, propName))
}

const buildToolPropertyTranslationKey = (toolId: string, propName: string) : string => {
  return toolId.split('-', 1)[0] + propName.charAt(0).toUpperCase() + propName.slice(1)
}

const solveToolPropertyTooltip = (toolId: string, propName: string, value: unknown) : string | null => {
  let tooltipKey = buildToolPropertyTranslationKey(toolId, propName) + 'Tooltip'
  let ret = t(tooltipKey)
  if (ret != tooltipKey) {
    return ret
  }
  if (value === undefined || value === null) {
    return null
  }
  const valStr = value.toString()
  tooltipKey = tooltipKey + valStr.charAt(0).toUpperCase() + valStr.slice(1)
  ret = t(tooltipKey)
  return ret != tooltipKey ? ret : null
}

const toolMessage = computed(() => {
  const toolMessageKey = buildToolPropertyTranslationKey(props.toolConfig.tool.id, 'toolMessage')
  const ret = t(toolMessageKey, { frontendUrl: window.location.origin })
  return ret != toolMessageKey ? ret : null
})

const hasFileProperties = computed(() =>
  Object.values(toolProperties.value).some(prop => isFileProperty(prop) || isFileArrayProperty(prop))
)

const saveButtonText = computed(() => {
  return hasFileProperties.value ? 'done' : 'save'
})

const isDoneDisabled = computed(() => {
  return saving.value || (hasFileProperties.value && !savedConfig.value && !allFilesRemoved.value)
})

const isFileArrayProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)
  return toolPropSchema?.type === 'array' && (js7(toolPropSchema?.items)?.$ref?.endsWith('/File') ?? false)
}

const isFileProperty = (toolProp: JSONSchema7Definition) : boolean => {
  return js7(toolProp)?.$ref?.endsWith('/File') ?? false
}

const isArrayEnumProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'array' && js7(toolPropSchema.items)?.enum !== undefined
}

const isStringEnumProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'string' && toolPropSchema.enum !== undefined
}

const isSecretStringProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'string' && toolPropSchema.writeOnly === true
}

const translateOptionLabel = (propName: string, option: string) : string => {
  return t(buildToolPropertyTranslationKey(props.toolConfig.tool.id, propName) + option.charAt(0).toUpperCase() + option.slice(1))
}

const onBeforeFileUpload = async (_: number) => {
  try {
    if (savedConfig.value != mutableConfig) {
      await api.configureAgentTool(props.toolConfig.agentId, new AgentToolConfig(props.toolConfig.toolId, mutableConfig))
      savedConfig.value = { ...mutableConfig }
      allFilesRemoved.value = false
    }
  } catch (error) {
    await handleError(error)
  }
}

const onAfterFileRemove = async (filesCount: number) => {
  try {
    if (filesCount === 0) {
      await api.removeAgentToolConfig(props.toolConfig.agentId, props.toolConfig.toolId)
      savedConfig.value = undefined
      allFilesRemoved.value = true
    }
  } catch (error) {
    await handleError(error)
  }
}

const onClose = () => {
  // when tool was configured (because files were uploaded), or the tool config was removed (because files were removed)
  // then we emit the update event so tools listing includes the new configured tool o removes existing tool
  // partialSave is a temporary solution, until we implement proper tool config drafts,
  // to properly refresh interface (removing the toolConfig) when existing tool config is changed
  // and saved in backend as a draft, but the configuration does not complete (eg: oauth cancellation)
  if (!props.toolConfig.config && savedConfig.value || props.toolConfig.config && !savedConfig.value || partialSave) {
    emit('update', new AgentToolConfig(props.toolConfig.toolId, mutableConfig))
  } else {
    emit('close')
  }
}

const saveToolConfig = async () => {
  saving.value = true
  clearValidationErrors()
  try {
    validateToolConfig()
    let ret = new AgentToolConfig(props.toolConfig.toolId, mutableConfig)
    // avoid saving tool when requires files and none have been uploaded
    if (savedConfig.value !== mutableConfig && !(Object.values(toolProperties.value).some(prop => isFileProperty(prop) || isFileArrayProperty(prop)) && !savedConfig.value)) {
      partialSave = true
      ret = await handleToolAuthRequestsIn(async () => {
        // resetting this flag since we want to allow retry saving tool config to re open oauth popup, or allow cancelling oauth
        // when oauth popup is closed or just ignored
        saving.value = true
        try {
          return await api.configureAgentTool(props.toolConfig.agentId, ret)
        } finally {
          // here is where we re enable save and cancel buttons in case oauth popup is showed or auth completed
          saving.value = false
        }
      }, api)
      partialSave = false
    }
    emit('update', ret)
  } catch (error) {
    console.error("Error saving tool config", error)
    if (error instanceof AuthenticationError) {
      validationErrors.value = t(error.errorCode)
    } else if (error instanceof ValidationErrors) {
      validationErrors.value = error.message
    } else if (error instanceof HttpError && error.status === 400) {
      validationErrors.value = t('invalidToolConfiguration', { contactEmail: contactEmail.value })
    } else {
      await handleError(error)
    }
  } finally {
    saving.value = false
  }
}

const validateToolConfig = () => {
  const ajv = new Ajv()
  addFormats(ajv)
  const toolConfig = props.toolConfig
  const schema = ajv.compile(removeFileProperties(toolConfig.tool.configSchema))
  const valid = schema(mutableConfig)
  if (!valid) {
    const errors = schema.errors?.map((error) => findValidationErrorMessage(error, toolConfig.tool.id))
    throw new ValidationErrors(errors?.join('\n') ?? '')
  }
}

const removeFileProperties = (schema: JSONSchema7) : JSONSchema7 => {
  const fileProperties = Object.entries(schema.properties ?? {}).filter(([_, value]) => isFileProperty(value) || isFileArrayProperty(value))
  return {
    ...schema,
    properties: Object.fromEntries(Object.entries(schema.properties ?? {}).filter(([_, value]) => !isFileProperty(value) && !isFileArrayProperty(value))),
    required: schema.required?.filter((property) => !fileProperties.some(([key, _]) => key === property))
  }
}

const findValidationErrorMessage = (error: ErrorObject, toolId: string) : string => {
  console.warn("Validation error", error)
  const property = error.instancePath.split('/').pop() ?? ''
  switch (error.keyword) {
    case 'required':
      return t('missingProperty', { property: translateToolPropertyName(toolId, error.params.missingProperty) })
    case 'format':
      return t('invalidPropertyFormat', { property: translateToolPropertyName(toolId, property), format: t(error.params.format + 'Format') })
    case 'minLength':
      return t('invalidPropertyMinLength', { property: translateToolPropertyName(toolId, property), minLength: error.params.limit }, error.params.limit)
    default:
      throw new Error(`UnknownValidationError: ${error.keyword}`)
  }
}

const clearValidationErrors = () => {
  validationErrors.value = null
}

class ValidationErrors extends Error {
  constructor(message: string) {
    super(message)
  }
}
</script>

<template>
  <FlexCard header-class="flex items-center justify-between">
    <template #header>
      <div class="flex w-full items-center justify-between">
        <div class="flex gap-2 items-center">
          <component :is="toolConfig.icon" />
          <span> | {{ toolConfig.name }}</span>
        </div>
        <SimpleButton @click="onClose" :disabled="saving">
          <IconX />
        </SimpleButton>
      </div>
    </template>
    <div class="flex flex-col gap-2 mb-4">
      <div v-if="toolMessage" class="text-content-muted text-sm tool-message mb-2" v-html="toolMessage"></div>
      <div v-for="propName in Object.keys(toolProperties)" :key="propName" class="form-field relative">
        <div v-if="isFileArrayProperty(toolProperties[propName]) || isFileProperty(toolProperties[propName])">
          <label :for="propName">{{ translateToolPropertyName(toolConfig.tool.id, propName) }}</label>
          <AgentToolFilesEditor
              :id="propName"
              :agent-id="toolConfig.agentId"
              :tool-id="toolConfig.tool.id"
              :configured-tool="savedConfig != undefined"
              :contact-email="contactEmail"
              :view-mode="viewMode"
              :allowed-extensions="(toolProperties[propName] as any)?.['x-allowed-extensions']"
              :max-files="isFileProperty(toolProperties[propName]) ? 1 : undefined"
              :on-before-file-upload="onBeforeFileUpload"
              :on-after-file-remove="onAfterFileRemove"/>
        </div>
        <div v-else-if="isBooleanProperty(toolProperties[propName])" class="flex items-center gap-2 text-sm">
          <ToggleSwitch
              v-model="mutableConfig[propName] as boolean"
              v-tooltip.bottom="solveToolPropertyTooltip(toolConfig.tool.id, propName, mutableConfig[propName])"
              :disabled="viewMode" />
          <span :class="{ 'text-content-muted': !mutableConfig[propName] }">{{ translateToolPropertyName(toolConfig.tool.id, propName) }}</span>
        </div>
        <div v-else-if="isArrayEnumProperty(toolProperties[propName])">
          <AgentToolConfigEnumPropertyEditor
              v-model="mutableConfig[propName] as string[]"
              :id="propName"
              :label="translateToolPropertyName(toolConfig.tool.id, propName)"
              :option-values="js7(js7(toolProperties[propName])!.items)!.enum as string[]"
              :option-labels="translateOptionLabel"
              :view-mode="viewMode"/>
        </div>
        <div v-else-if="isStringEnumProperty(toolProperties[propName])" class="flex flex-col gap-1">
          <label :for="propName" v-tooltip.bottom="solveToolPropertyTooltip(toolConfig.tool.id, propName, mutableConfig[propName])">{{ translateToolPropertyName(toolConfig.tool.id, propName) }}</label>
          <Select
              v-model="mutableConfig[propName]"
              :id="propName"
              :options="(js7(toolProperties[propName])!.enum as string[]).map(val => ({ label: translateOptionLabel(propName, val), value: val }))"
              optionLabel="label"
              optionValue="value"
              class="w-1/2 my-1"
              v-tooltip.bottom="solveToolPropertyTooltip(toolConfig.tool.id, propName, mutableConfig[propName])"
              :disabled="viewMode"
              @change="clearValidationErrors"/>
        </div>
        <div v-else-if="!isBooleanProperty(toolProperties[propName]) && areDependenciesMet(toolProperties[propName])" class="flex flex-col gap-1">
          <label :for="propName">{{ translateToolPropertyName(toolConfig.tool.id, propName) }}</label>
          <InteractiveInput
              v-model="mutableConfig[propName] as string"
              :id="propName"
              :type="isSecretStringProperty(toolProperties[propName]) ? 'password' : 'text'"
              :disabled="viewMode"/>
        </div>
      </div>
      <div v-if="validationErrors" class="text-error-alt validation-errors" v-html="validationErrors"></div>
      <div class="flex justify-end gap-2" v-if="!viewMode">
        <SimpleButton v-if="!(hasFileProperties)" @click="onClose" variant="secondary" shape="square" :disabled="saving">
          {{ t('cancel') }}
        </SimpleButton>
        <SimpleButton @click="saveToolConfig" variant="primary" shape="square" :disabled="isDoneDisabled">
          {{ t(saveButtonText) }}
        </SimpleButton>
      </div>
    </div>
  </FlexCard>
</template>

<style>
@import '@/assets/styles.css';

.validation-errors a {
  @apply text-abstracta inline;
}

.tool-message a {
  @apply inline;
}
</style>

<i18n lang="json">
  {
    "en": {
      "save": "Save",
      "done": "Done",
      "cancel": "Cancel",
      "close": "Close",
      "authenticationWindowBlocked": "The authentication popup could not be opened. Please check that popups are allowed for this site and try again.",
      "authenticationCancelled": "The authentication was cancelled. Please complete the authentication process to configure this tool.",
      "authenticationAccessDenied": "The authentication was denied by the server. Please verify that you actually have the permissions necessary to use it.",
      "missingProperty": "A value is required for '{property}'. Please provide one.",
      "invalidPropertyFormat": "Provided '{property}' is not a valid {format}. Please, review the value and try again.",
      "invalidPropertyMinLength": "Provided '{property}' is shorter than {minLength} @:{'character'}. Please, review the value and try again.",
      "character": "character | characters",
      "uriFormat": "URL",
      "invalidToolConfiguration": "Tool configuration failed. Please review the configuration and try again. If the problem persists, contact {'<'}a href='mailto:{contactEmail}?subject=Tero%20Error'>{contactEmail}{'<'}/a>.",
      "docsToolMessage": "Allows to use information from uploaded files in agent responses.{'<'}br/>To avoid unnecessary costs and incorrect agent answers, review that the documents and their contents are relevant to this agent.",
      "docsFiles": "Files",
      "docsAdvancedFileProcessing": "Process new PDFs with advanced AI (higher budget usage)",
      "docsAdvancedFileProcessingTooltipFalse": "Basic processing uses a simple algorithm to extract the content of the file. In general it is less accurate but it is faster and consumes less budget. \n\nNote: This option will only apply to new uploaded files.",
      "docsAdvancedFileProcessingTooltipTrue": "Advanced processing uses AI to extract the content of the file. In general it is more accurate but it consumes more budget and it may take longer to process. \n\nNote: This option will only apply to new uploaded files.",
      "mcpToolMessage": "Enable using tools provided by an MCP server for this agent.{'<'}br/>{'<'}b>Only use servers you trust.{'<'}/b>",
      "mcpServerUrl": "Server URL",
      "mcpAuthType": "Auth Type",
      "mcpAuthTypeOauth": "OAuth",
      "mcpAuthTypeBearerToken": "Bearer Token",
      "mcpBearerToken": "Bearer Token",
      "jiraToolMessage": "Allows to integrate this agent with Jira.{'<'}br/>To configure this tool you need a configured Jira OAuth app with redirect URL {'<'}u>{frontendUrl}/tools/jira/oauth-callback{'<'}/u>.{'<'}br/> Check {'<'}a href='https://developer.atlassian.com/cloud/confluence/oauth-2-3lo-apps/#enabling-oauth-2-0--3lo-' target='_blank'>this guide{'<'}/a> for more details.",
      "jiraClientId": "Client ID",
      "jiraClientSecret": "Client secret",
      "jiraScope": "Scopes",
      "jiraScopeRead:jira-work": "Read",
      "jiraScopeWrite:jira-work": "Write",
      "jiraScopeRead:jira-user": "User Info",
      "webToolMessage": "Enable web search and URL content extraction for this agent.",
      "browserToolMessage": "Enable executing actions on a remote chromium browser.{'<'}br/>You can check {'<'}a href='https://hub.docker.com/r/mcp/playwright#available-tools-21' target='_blank'>here{'<'}/a> the list of available actions."
    },
    "es": {
      "save": "Guardar",
      "done": "Listo",
      "cancel": "Cancelar",
      "close": "Cerrar",
      "authenticationWindowBlocked": "No se pudo abrir la ventana de autenticación. Por favor, verifica que las ventanas emergentes estén permitidas para este sitio y vuelve a intentarlo.",
      "authenticationCancelled": "La autenticación fue cancelada. Por favor, complete el proceso de autenticación para configurar esta herramienta.",
      "authenticationAccessDenied": "La autenticación fue denegada por el servidor. Por favor, verifica que tengas los permisos necesarios para usarlo.",
      "missingProperty": "Se requiere un valor para '{property}'. Por favor proporciona uno.",
      "invalidPropertyFormat": "El valor proporcionado para '{property}' no es válido. Por favor, revise el valor y vuelve a intentarlo.",
      "invalidPropertyMinLength": "El valor proporcionado para '{property}' es más corto que {minLength} @:{'character'}. Por favor, revise el valor y vuelve a intentarlo.",
      "character": "caracter | caracteres",
      "uriFormat": "URL",
      "invalidToolConfiguration": "La configuración del tool falló. Por favor revise la configuración y vuelve a intentarlo. Si el problema persiste, contacta a {'<'}a href='mailto:{contactEmail}?subject=Tero%20Error'>{contactEmail}{'<'}/a>.",
      "docsToolMessage": "Permite usar el contenido de documentos como base de conocimiento de este agente.{'<'}br/>Para evitar costos innecesarios y respuestas incorrectas del agente, revisa que los documentos y sus contenidos sean relevantes para este agente.",
      "docsFiles": "Archivos",
      "docsAdvancedFileProcessing": "Procesar nuevos PDF con AI avanzada (mayor consumo de budget)",
      "docsAdvancedFileProcessingTooltipFalse": "El procesamiento básico utiliza un algoritmo simple para extraer el contenido del archivo. En general es menos preciso pero es más rápido y consume menos presupuesto. \n\nNota: Esta opción se aplicará unicamente a los nuevos archivos subidos.",
      "docsAdvancedFileProcessingTooltipTrue": "El procesamiento avanzado utiliza IA para extraer el contenido del archivo. En general es más preciso pero consume más presupuesto y puede tardar más en procesarse.\n\nNota: Esta opción se aplicará unicamente a los nuevos archivos subidos.",
      "mcpToolMessage": "Habilita el uso de herramientas de un servidor MCP para este agente.{'<'}br/>{'<'}b>Solo usa servidores que confíes.{'<'}/b>",
      "mcpServerUrl": "URL del servidor",
      "mcpAuthType": "Tipo de autenticación",
      "mcpAuthTypeOauth": "OAuth",
      "mcpAuthTypeBearerToken": "Bearer Token",
      "mcpBearerToken": "Bearer Token",
      "jiraToolMessage": "Permite integrar este agente con Jira.{'<'}br/>Para configurar esta herramienta necesitas configurar una aplicación OAuth en Jira con URL de redirección {'<'}u>{frontendUrl}/tools/jira/oauth-callback{'<'}/u>.{'<'}br/> Revisa {'<'}a href='https://developer.atlassian.com/cloud/confluence/oauth-2-3lo-apps/#enabling-oauth-2-0--3lo-' target='_blank'>esta guía{'<'}/a> para más detalles.",
      "jiraClientId": "ID del cliente",
      "jiraClientSecret": "Secreto del cliente",
      "jiraScope": "Scopes",
      "jiraScopeRead:jira-work": "Leer",
      "jiraScopeWrite:jira-work": "Escribir",
      "jiraScopeRead:jira-user": "Info. del usuario",
      "webToolMessage": "Habilita búsqueda web y extracción de contenido desde URL para este agente.",
      "browserToolMessage": "Habilita la ejecución de acciones en un navegador chromium remoto.{'<'}br/>Puedes revisar {'<'}a href='https://hub.docker.com/r/mcp/playwright#available-tools-21' target='_blank'>aquí{'<'}/a> la lista de acciones disponibles."
    }
  }
</i18n>
