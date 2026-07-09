<script setup lang="ts">
import { ref, computed, getCurrentInstance } from 'vue';
import type { MenuItem } from 'primevue/menuitem';
import { IconLock } from '@tabler/icons-vue';
import { useI18n } from 'vue-i18n';

const props = defineProps<{
  agentTeam?: string;
  isCollapsed?: boolean;
  items?: MenuItem[];
  active?: boolean;
  canViewAgentInfo?: boolean;
}>();

const { t } = useI18n();

const menu = ref();
const isMenuOpen = ref(false);

const emit = defineEmits(['toggleActive', 'closeMenu']);

const shouldShowInSidebar = computed(() => {
  const instance = getCurrentInstance();
  return instance?.parent?.type?.__name?.includes('Sidebar') || false;
});

const toggle = (event: Event) => {
  menu.value?.toggle(event);
  emit('toggleActive');
};

const closeMenu = () => {
  emit('closeMenu');
}

defineExpose({
  menu,
  isMenuOpen,
  toggle
});
</script>

<template>
  <div v-click-outside="closeMenu">
    <div v-if="shouldShowInSidebar"
         class="transition-opacity border-l-[1px]"
         :class="[
           !isCollapsed ? 'mx-2' : '',
           isMenuOpen ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
         ]"
         @click.stop.prevent="toggle">
      <IconDots :class="['text-content-muted', !isCollapsed ? 'ml-3' : '']" />
    </div>
    <SimpleButton v-else
                  :class="`rounded-xl ${!active ? 'dark:bg-content-muted' : ''}`"
                  variant="muted"
                  size="small"
                  @click.stop.prevent="toggle">
      <IconDots stroke-width="2" :class="!active ? 'text-content-muted dark:text-gray-800' : 'text-abstracta'" />
    </SimpleButton>

    <Menu
      ref="menu"
      :model="items ?? []"
      :popup="true"
      @show="isMenuOpen = true"
      @hide="isMenuOpen = false"
      @keydown.esc="closeMenu">
      <template #item="{ item }">
        <MenuItemTemplate :item="item"/>
      </template>
      <template #end>
        <span v-if="!canViewAgentInfo" class="block bg-surface-muted text-content-muted text-center text-sm font-semibold truncate px-2 py-1 w-[calc(100%+0.5rem)] ml-[-0.25rem] mr-[-0.25rem]">
          {{ t('protected') }}
        </span>
        <span v-if="agentTeam" class="block bg-abstracta text-center text-white text-sm font-semibold truncate px-2 py-1 rounded-b-xl w-[calc(100%+0.5rem)] mb-[-0.25rem] ml-[-0.25rem] mr-[-0.25rem]">
          {{ agentTeam }}
        </span>
      </template>
    </Menu>
  </div>
</template>

<i18n lang="json">
  {
    "en": {
      "protected": "Protected Agent"
    },
    "es": {
      "protected": "Agente protegido"
    }
  }
</i18n>
