<script setup lang="ts">
import { useI18n } from 'vue-i18n';
import { ApiService, Agent, AgentToolConfig, LlmModel, LlmModelType } from '@/services/api';
import { useChatStore } from '@/composables/useChatStore';
import { useAgentStore } from '@/composables/useAgentStore';
import { useErrorHandler } from '@/composables/useErrorHandler';
import moment from 'moment';
import { IconCopyPlus, IconEditCircle, IconX, IconArrowUpRight } from '@tabler/icons-vue';
import { computed, ref, watch } from 'vue';

const { t } = useI18n();
const { newChat } = useChatStore();
const { addAgent, cloneAgent, configureAgent } = useAgentStore();
const { handleError } = useErrorHandler();

const api = new ApiService();

const props = defineProps<{
  agent: Agent;
  showModal: boolean;
}>();

const agent = ref<Agent>(props.agent);
const tools = ref<AgentToolConfig[]>([]);
const models = ref<LlmModel[]>([]);
const isLoading = ref(false);

const currentModel = computed(() => {
  if (!agent.value?.modelId || !models.value.length) return null;
  return models.value.find((m) => m.id === agent.value?.modelId);
});

const modelType = computed(() => currentModel.value?.modelType);

const emit = defineEmits<{
  (e: 'close'): void;
}>();

const startChat = async () => {
  try {
    await addAgent(agent.value);
    await newChat(agent.value);
    onClose();
  } catch (e) {
    handleError(e);
  }
};

const onCloneAgent = async () => {
  try {
    await cloneAgent(agent.value.id);
  } catch (e) {
    handleError(e);
  }
}

const onEditAgent = async () => {
  try {
    await configureAgent(agent.value.id);
  } catch (error) {
    handleError(error);
  }
}

const onClose = () => {
  emit('close');
}

const canView = computed(() => agent.value.canEdit || !agent.value.isProtected);

watch(() => [props.showModal, props.agent.id],
  async ([newShowModal]) => {
    if (newShowModal && canView.value) {
      try {
        isLoading.value = true;
        agent.value = await api.findAgentById(props.agent.id);
        tools.value = await api.findAgentToolConfigs(agent.value.id);
        models.value = await api.findModels();
      } catch (e) {
        handleError(e);
      } finally {
        isLoading.value = false;
      }
    }
  },
);
</script>

<template>
  <Dialog :visible="showModal" @update:visible="onClose" :modal="true" :draggable="false" :resizable="false" :closable="false" class="w-220" :showHeader="false" :dismissableMask="true">
      <DiscoverAgentInfoSkeleton v-if="canView && isLoading" />
      <div v-if="!isLoading" class="flex flex-col gap-4 relative">
        <div class="flex items-center gap-2 justify-between w-full pb-5 pt-5 border-b">
          <div class="flex flex-row gap-2 items-center flex-1 min-w-0">
            <AgentAvatar :agent="agent" size="large"/>
            <div class="flex flex-col gap-1 min-w-0">
              <span class="font-semibold text-2xl truncate min-w-0 max-w-full block">{{ agent.name }}</span>
              <div class="flex gap-1 text-content-muted text-xs w-fit">
                <span v-if="agent.user?.name" class="truncate">{{ t('by') }} <span class="font-semibold">{{ agent.user.name }}</span></span>
                <span v-if="agent.user?.name">|</span>
                <span class="truncate">
                  {{ t('updatedAt') }}
                  <span class="font-semibold">{{ moment(agent.lastUpdate).format('DD MMM YYYY') }}</span>
                </span>
              </div>
            </div>
          </div>
          <div class="flex flex-row gap-4 justify-start h-full self-start flex-shrink-0 items-center">
            <SimpleButton v-if="canView" size="small" shape="square" class="px-3" @click="onCloneAgent">
              <IconCopyPlus/> {{ t('cloneButtonLabel') }}
            </SimpleButton>
            <SimpleButton v-if="agent.canEdit" size="small" shape="square" class="px-3 whitespace-nowrap" @click="onEditAgent">
              <IconEditCircle/> {{ t('editButtonLabel') }}
            </SimpleButton>
            <SimpleButton size="small" shape="rounded" class="" @click="onClose" >
              <IconX class="w-6 h-6" />
            </SimpleButton>
          </div>
        </div>
        <span :class="{ 'text-sm text-content-muted': !agent.description }">{{ agent.description || t('noDescription') }}</span>
        <div v-if="canView" class="flex flex-col gap-4">
          <div class="flex flex-row gap-2 bg-surface-muted p-2 px-8 rounded-md items-center justify-between">
            <div class="flex flex-row gap-4 items-center">
              <span class="text-sm font-semibold">{{ t('modelLabel') }}</span>
              <span class="text-m p-1 px-4 rounded-md bg-surface font-semibold flex flex-row gap-2 items-center">
                <VendorLogo :vendor="currentModel!.modelVendor"/>
                <span>{{ currentModel?.name }}</span>
              </span>
            </div>
            <div class="flex flex-row gap-4 items-center">
              <span class="text-sm font-semibold">{{LlmModelType.CHAT === modelType ? t('temperatureLabel') : t('reasoningEffortLabel')}}</span>
              <span class="text-m p-1 px-4 rounded-md bg-surface font-semibold lowercase first-letter:uppercase">
                <AgentModelConfig :agent="agent" :modelType="modelType!"/>
              </span>
            </div>
          </div>
          <div class="form-field relative gap-2">
            <span class="text-m font-semibold">{{ t('systemPromptLabel') }}</span>
            <div class="text-m whitespace-pre-line p-4 border rounded-md max-h-50 overflow-y-auto">
              {{ agent.systemPrompt }}
            </div>
          </div>
          <div class="form-field relative gap-2">
            <span class="text-m font-semibold" >{{ t('toolsLabel') }}</span>
            <div class="flex flex-row gap-2 items-center justify-between" :class="{ 'justify-end': !canView }">
              <div class="form-field relative flex-1">
                <AgentToolConfigsEditor :agent-id="agent.id" :toolConfigs="tools" :viewMode="true"/>
              </div>
            </div>
          </div>
        </div>
        <DiscoverAgentInfoProtected v-else />
        <SimpleButton size="small" shape="square" class="px-3 w-fit self-end" @click="startChat" variant="primary">
          {{ t('startChatButtonLabel') }}
          <IconArrowUpRight />
        </SimpleButton>
      </div>
  </Dialog>
</template>

<i18n lang="json">
  {
    "en": {
      "startChatButtonLabel": "Use now",
      "by": "by",
      "updatedAt": "Last updated",
      "cloneButtonLabel": "Clone",
      "editButtonLabel": "Edit agent",
      "systemPromptLabel": "Instructions",
      "modelLabel": "Model",
      "toolsLabel": "Tools",
      "noDescription": "No description",
      "temperatureLabel": "Temperature",
      "reasoningEffortLabel": "Reasoning",
      "viewToolTooltip": "View tool"
    },
    "es": {
      "startChatButtonLabel": "Usar ahora",
      "by": "por",
      "updatedAt": "Actualizado el",
      "cloneButtonLabel": "Clonar",
      "editButtonLabel": "Editar agente",
      "systemPromptLabel": "Instrucciones",
      "modelLabel": "Modelo",
      "toolsLabel": "Herramientas",
      "noDescription": "No hay descripción",
      "temperatureLabel": "Temperatura",
      "reasoningEffortLabel": "Razonamiento",
      "viewToolTooltip": "Ver herramienta"
    }
  }
</i18n>
