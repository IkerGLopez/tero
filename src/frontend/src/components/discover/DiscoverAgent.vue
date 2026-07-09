<script setup lang="ts">
import { useI18n } from 'vue-i18n';
import type { Agent } from '@/services/api';
import { useChatStore } from '@/composables/useChatStore';
import { useAgentStore } from '@/composables/useAgentStore';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { ref } from 'vue';
import { IconInfoCircle, IconUsers, IconLock, IconArrowUpRight } from '@tabler/icons-vue';
import DiscoverAgentInfo from './DiscoverAgentInfo.vue';

const { t } = useI18n();
const { newChat } = useChatStore();
const { addAgent } = useAgentStore();
const { handleError } = useErrorHandler();

const props = defineProps<{
  agent: Agent;
  showTeamBadge?: boolean;
}>();

const showAgentInfoModal = ref<boolean>(false);

const startChat = async () => {
  try {
    await addAgent(props.agent);
    await newChat(props.agent);
  } catch (e) {
    handleError(e);
  }
};
const showAgentInfo = () => {
  showAgentInfoModal.value = !showAgentInfoModal.value;
};
</script>

<template>
  <div class="grid-card px-3 py-2.5 border hover:shadow-sm cursor-pointer" @click="startChat">
    <div v-if="showTeamBadge && agent.team" class="absolute top-[-0.75rem] right-3">
      <SimpleTag :text="agent.team.name" />
    </div>
    <div class="flex flex-row items-center justify-between">
      <div class="flex flex-row gap-2 items-center text-base min-w-0">
        <AgentAvatar :agent="agent" />
        <div class="flex flex-col min-w-0 flex-1">
          <span class="truncate block">{{ agent.name }}</span>
          <span v-if="agent.user && agent.team" class="text-content-muted text-xs truncate">{{ t('by') }} <span class="font-medium">{{ agent.user.name }}</span></span>
        </div>
      </div>
    </div>

    <div v-if="agent.description" class="mt-2 line-clamp-3 text-sm">
      {{ agent.description }}
    </div>

    <div class="mt-auto text-sm flex flex-row gap-2 items-center justify-between">
      <div class="flex gap-2">
        <SimpleButton shape="square" size="small" variant="secondary" @click.stop="showAgentInfo" v-tooltip.bottom="t('viewDetailsTooltip')">
          <IconInfoCircle class="w-5 h-5" />
        </SimpleButton>
        <a class="gap-1">
          {{ t('callToAction') }}
          <IconArrowUpRight class="w-5 h-5" />
        </a>
      </div>
      <div class="text-content-muted">
        <template v-if="agent.team">
          <div v-tooltip.bottom="t('activeUsersTooltip')" class="flex flex-row gap-2 items-center">
            <IconUsers class="w-5 h-5" />
            <span class="translate-y-0.5">{{ agent.activeUsers || 0 }}</span>
          </div>
        </template>
        <template v-else>
          <IconLock />
        </template>
      </div>
    </div>
  </div>
  <DiscoverAgentInfo :agent="agent" :showModal="showAgentInfoModal" @close="showAgentInfo" />
</template>

<i18n>
  {
    "en": {
      "callToAction": "Use now",
      "activeUsersTooltip": "Number of users that used this agent in the last 30 days.",
      "by": "by",
      "viewDetailsTooltip": "View details"
    },
    "es": {
      "callToAction": "Usar ahora",
      "activeUsersTooltip": "Número de usuarios que han utilizado este agente en los últimos 30 días.",
      "by": "por",
      "viewDetailsTooltip": "Ver detalles"
    }
  }
</i18n>
