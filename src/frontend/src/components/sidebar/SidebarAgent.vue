<script setup lang="ts">
import { ref, computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { useAgentStore } from '@/composables/useAgentStore';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { useChatStore } from '@/composables/useChatStore';
import { AnimationEffect } from '../../../../common/src/utils/animations';
import type { Agent } from '@/services/api';
import { useSidebar } from '@/composables/useSidebar';
import { IconMessage2Plus, IconEditCircle, IconTrashX, IconCopyPlus, IconInfoCircle } from '@tabler/icons-vue';

const { isSidebarCollapsed } = useSidebar();
const { t } = useI18n();
const { newChat } = useChatStore();
const { removeAgent, cloneAgent } = useAgentStore();
const { handleError } = useErrorHandler();
const { configureAgent } = useAgentStore();
const showConfirmation = ref(false);
const showAgentInfoModal = ref(false);
const isHovered = ref(false);
const menu = ref();

const props = defineProps<{
  agent: Agent
}>()

const onNewChat = async () => {
  try {
    await newChat(props.agent);
  } catch (e) {
    handleError(e);
  }
};

const handleRemoveAgent = () => {
  showConfirmation.value = true;
}

const onRemoveAgent = async () => {
  try {
    await removeAgent(props.agent.id);
  } catch (e) {
    handleError(e);
  }
}

const handleEditAgent = async () => {
  configureAgent(props.agent.id);
}

const handleCloneAgent = async () => {
  try {
    await cloneAgent(props.agent.id);
  } catch (error) {
    handleError(error);
  }
}

const handleShowAgentInfo = () => {
  showAgentInfoModal.value = !showAgentInfoModal.value;
};

const canViewAgentInfo = computed(() => props.agent.canEdit || !props.agent.isProtected);

</script>

<template>
  <SidebarItem
    v-if="!showConfirmation"
    :is-collapsed="isSidebarCollapsed"
    @click="onNewChat()"
    @mouseenter="isHovered = true"
    @mouseleave="isHovered = false"
    @menu-toggle="(event: Event) => menu?.toggle(event)"
    :class="isSidebarCollapsed ? 'justify-center' : ''"
    v-tooltip.bottom="{value: agent.name, showDelay: 1000}">
    <div class="flex items-center" :class="isSidebarCollapsed ? 'justify-center w-full' : 'gap-2'">
      <div class="flex-shrink-0 flex items-center relative">
        <AgentAvatar :agent="agent" :show-shared-status="true" />
      </div>
      <span v-if="!isSidebarCollapsed" class="line-clamp-1 break-all min-w-0 flex-1">
        {{ agent.name }}
      </span>
    </div>
    <template #actions v-if="!isSidebarCollapsed && (isHovered || menu?.isMenuOpen)">
      <AgentChatMenu
        ref="menu"
        :agent-team="agent.team?.name"
        :is-collapsed="isSidebarCollapsed"
        :can-view-agent-info="canViewAgentInfo"
        :items="[
          {
            label: t('newChatTooltip'),
            tablerIcon: IconMessage2Plus,
            command: onNewChat
          },
          {
            label: t('agentInfoTooltip'),
            tablerIcon: IconInfoCircle,
            command: handleShowAgentInfo
          },
          ...(agent.canEdit ? [{
            label: t('editAgentTooltip'),
            tablerIcon: IconEditCircle,
            command: handleEditAgent
          }] : []),
          ...(canViewAgentInfo ? [{
            label: t('cloneAgentTooltip'),
            tablerIcon: IconCopyPlus,
            command: handleCloneAgent
          }] : []),
          {
            label: t('removeAgentTooltip'),
            tablerIcon: IconTrashX,
            command: handleRemoveAgent
          }]"
      />
    </template>
  </SidebarItem>

  <Animate v-if="showConfirmation" :effect="AnimationEffect.QUICK_SLIDE_DOWN">
    <ItemConfirmation
      :tooltip="t('removeAgentConfirmation')"
      @confirm="onRemoveAgent"
      @cancel="() => showConfirmation = false"
    />
  </Animate>
  <DiscoverAgentInfo :agent="agent" :showModal="showAgentInfoModal" @close="handleShowAgentInfo" />
</template>

<i18n>
  {
    "en": {
      "removeAgentTooltip": "Remove agent",
      "removeAgentConfirmation": "Remove agent?",
      "newChatTooltip": "New chat",
      "editAgentTooltip": "Edit agent",
      "cloneAgentTooltip": "Clone agent",
      "agentInfoTooltip": "View details"
    },
    "es": {
      "removeAgentTooltip": "Quitar agente",
      "removeAgentConfirmation": "¿Quitar agente?",
      "newChatTooltip": "Nuevo chat",
      "editAgentTooltip": "Editar agente",
      "cloneAgentTooltip": "Clonar agente",
      "agentInfoTooltip": "Ver detalles"
    }
  }
</i18n>
