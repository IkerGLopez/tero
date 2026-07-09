<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { ApiService } from '@/services/api'
import { AgentToolConfig, type AgentTool } from '@/services/api'
import { useErrorHandler } from '@/composables/useErrorHandler'
import { type Icon } from '@tabler/icons-vue'
import { EditingToolConfig } from './AgentToolConfigEditor.vue'
import { IconEditCircle, IconEye, IconX } from '@tabler/icons-vue'
import { findToolIcon, buildToolConfigName } from '@tero/common/utils/toolConfig.js'
import GroupedSelectPanel, { type GroupedSelectPanelOptionGroup } from '../common/GroupedSelectPanel.vue'

const { t } = useI18n()
const api = new ApiService()
const { handleError } = useErrorHandler()

const props = defineProps<{
  agentId: number
  toolConfigs: AgentToolConfig[]
  viewMode?: boolean
}>()

const emit = defineEmits<{
  (e: 'update'): void
}>()

const availableTools = ref<AgentTool[]>([])
const editingToolConfig = ref<EditingToolConfig | null>(null)
const deletingToolConfig = ref<string | null>(null)

onMounted(async () => {
  try {
    availableTools.value = await api.findAgentTools()
  } catch (error) {
    handleError(error)
  }
})

const onConfigureNewTool = (tool: AgentTool, toolName: string, toolIcon: Icon) => {
  editingToolConfig.value = new EditingToolConfig(props.agentId, tool, toolName, toolIcon)
}

const onSaveToolConfig = async () => {
  onCloseToolConfig()
  emit('update')
}

const onCloseToolConfig = () => {
  editingToolConfig.value = null
}

const onEditToolConfig = (toolConfig: AgentToolConfig) => {
  const tool = getToolById(toolConfig.toolId)
  editingToolConfig.value = new EditingToolConfig(props.agentId, tool!, buildToolConfigName(tool.id), findToolIcon(tool.id), toolConfig)
}

const hasConfigurableProperties = (toolId: string): boolean => {
  const tool = getToolById(toolId)
  return !!(tool?.configSchema?.properties && Object.keys(tool.configSchema.properties).length > 0)
}

const getToolById = (toolId: string): AgentTool => {
  return availableTools.value.find((tool) => (tool.id.endsWith('-*') && toolId.startsWith(tool.id.split('-', 1)[0])) || tool.id === toolId)!
}

const onDeleteToolConfig = (toolConfig: AgentToolConfig) => {
  deletingToolConfig.value = toolConfig.toolId
}

const onConfirmDelete = async (toolId: string) => {
  deletingToolConfig.value = null
  try {
    await api.removeAgentToolConfig(props.agentId, toolId)
    emit('update')
  } catch (error) {
    handleError(error)
  }
}

const onCancelDelete = () => {
  deletingToolConfig.value = null
}

const searchQuery = ref('')
const containerRef = ref<HTMLDivElement>()
const groupedSelectPanelRef = ref<InstanceType<typeof GroupedSelectPanel>>()
const showAllTools = ref(false)

const initialToolsLimit = 4

const groupedToolOptions = computed<GroupedSelectPanelOptionGroup[]>(() => {
  const q = searchQuery.value.trim().toLowerCase()
  let list = availableTools.value.filter((tool) => !props.toolConfigs.some((config) => config.toolId === tool.id))
  if (q) {
    list = list.filter((tool) => buildToolConfigName(tool.id).toLowerCase().includes(q) || tool.description?.toLowerCase().includes(q))
  }
  return list.map((tool) => ({
    id: tool.id,
    name: buildToolConfigName(tool.id),
    description: tool.description
  }))
})

const displayedToolOptions = computed(() => {
  const list = groupedToolOptions.value
  if (showAllTools.value || searchQuery.value.trim()) return list
  if (list.length <= initialToolsLimit + 1) return list
  return list.slice(0, initialToolsLimit)
})

const showLoadMoreTools = computed(() => !showAllTools.value && groupedToolOptions.value.length > initialToolsLimit + 1 && !searchQuery.value.trim())

const onShowAllTools = () => {
  groupedSelectPanelRef.value?.onShowDropdown()
}

const onToolChange = (toolId: string) => {
  const tool = getToolById(toolId)
  onConfigureNewTool(tool, buildToolConfigName(tool.id), findToolIcon(tool.id))
  onShowAllTools()
}

const onToolSearch = (value: string) => {
  searchQuery.value = value
  if (value?.trim()) {
    showAllTools.value = true
  }
}
</script>

<template>
  <div class="flex items-center justify-between w-full mb-2 relative" v-if="!viewMode" ref="containerRef">
    <label class="!text-sm"> {{ t('toolsLabel') }} </label>
    <SimpleButton size="small" shape="square" class="px-3" @click="onShowAllTools"> <IconPlus /> {{ t('addTool') }} </SimpleButton>
    <GroupedSelectPanel ref="groupedSelectPanelRef" :search-placeholder="t('searchPlaceholder')" :container="containerRef" :show-load-more="showLoadMoreTools" height="23rem" @search="onToolSearch" @load-more="showAllTools = true">
      <template #content>
        <div v-if="groupedToolOptions.length > 0" class="flex flex-col w-full">
          <div v-for="(option, index) in displayedToolOptions" :key="option.id" class="py-2" :class="[index < displayedToolOptions.length - 1 && 'border-b-2 border-dotted']">
            <div class="flex hover:bg-surface-muted rounded-2xl p-4 px-2 cursor-pointer gap-4" @click="onToolChange(option.id)">
              <div class="flex flex-row gap-2 w-30 shrink-0 items-center h-fit">
                <component :is="findToolIcon(option.id)" />
                <span class="font-semibold">{{ option.name }}</span>
              </div>
              <span v-if="option.description" class="flex-1">{{ option.description }}</span>
            </div>
          </div>
        </div>
        <div v-else class="text-sm text-content-muted text-center py-2 min-h-[30vh] flex items-center justify-center">
          {{ t('noOptionsPlaceholder') }}
        </div>
      </template>
    </GroupedSelectPanel>
  </div>
  <div class="flex flex-col gap-2">
    <div v-if="toolConfigs.length === 0" class="text-sm text-content-muted">
      <div>{{ t('noTools') }}</div>
    </div>
    <div v-else class="flex flex-row flex-wrap gap-2 w-0 min-w-full">
      <div v-for="tool in toolConfigs" :key="tool.toolId" class="relative rounded-xl border-1">
        <!-- using absolute position flex and inset in confirmation and invisible in item with py to keep item confirmation the same size as element -->
        <ItemConfirmation v-if="deletingToolConfig === tool.toolId" class="shadow-none !m-0 border-none flex-1 absolute inset-0" :tooltip="t('deleteToolConfigConfirmation')" @confirm="() => onConfirmDelete(tool.toolId)" @cancel="onCancelDelete" />
        <div class="flex justify-between items-center py-3 px-2 min-w-45" :class="{ invisible: deletingToolConfig === tool.toolId }">
          <div class="flex flex-row items-center gap-2 pr-2">
            <div>
              <component :is="findToolIcon(tool.toolId)" />
            </div>
            <span>{{ buildToolConfigName(tool.toolId) }}</span>
          </div>
          <div class="flex flex-row justify-center items-center border-l pl-2 gap-2">
            <SimpleIcon interactive v-if="hasConfigurableProperties(tool.toolId)" @click="onEditToolConfig(tool)" v-tooltip.bottom="viewMode ? t('viewToolConfig') : t('editToolConfig')" :icon="viewMode ? IconEye : IconEditCircle" />
            <SimpleIcon interactive @click="onDeleteToolConfig(tool)" v-tooltip.bottom="t('deleteToolConfig')" :icon="IconX" class="hover:text-error-alt" v-if="!viewMode" />
          </div>
        </div>
      </div>
    </div>
  </div>
  <Dialog :visible="editingToolConfig !== null" @update:visible="onCloseToolConfig" :modal="true" :draggable="false" :resizable="false" :closable="false" class="basic-dialog" maximizable :style="{ width: '50rem' }">
    <AgentToolConfigEditor v-if="editingToolConfig" :toolConfig="editingToolConfig" @update="onSaveToolConfig" @close="onCloseToolConfig" :viewMode="viewMode" />
  </Dialog>
</template>

<i18n lang="json">
{
  "en": {
    "toolsLabel": "Tools",
    "addTool": "Add tool",
    "noTools": "Add tools the agent can use when responding.",
    "editToolConfig": "Edit",
    "viewToolConfig": "View",
    "deleteToolConfig": "Remove",
    "deleteToolConfigConfirmation": "Remove?",
    "searchPlaceholder": "Search tools...",
    "noOptionsPlaceholder": "No tools found"
  },
  "es": {
    "toolsLabel": "Herramientas",
    "addTool": "Agregar herramienta",
    "noTools": "Agrega herramientas que este agente puede usar cuando responda.",
    "editToolConfig": "Editar",
    "viewToolConfig": "Ver",
    "deleteToolConfig": "Quitar",
    "deleteToolConfigConfirmation": "¿Quitar?",
    "searchPlaceholder": "Buscar herramientas...",
    "noOptionsPlaceholder": "No se encontraron herramientas"
  }
}
</i18n>
