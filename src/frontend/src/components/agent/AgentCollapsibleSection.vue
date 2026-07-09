<script lang="ts" setup>
import { IconCaretDownFilled, IconCaretUpFilled } from '@tabler/icons-vue'

defineProps<{
  title: string
  collapsed: boolean
}>()

const emit = defineEmits<{
  (e: 'update:collapsed', value: boolean): void
}>()
</script>

<template>
  <Panel class="!border-none" toggleable :collapsed="collapsed" @update:collapsed="emit('update:collapsed', $event)">
    <template #header>
      <div class="flex-1 cursor-pointer" @click="emit('update:collapsed', !collapsed)">
        <slot name="header" :title="title">
          <h4 class="!text-sm font-semibold uppercase tracking-[0.2em]">
            {{ title }}
          </h4>
        </slot>
      </div>
    </template>
    <template #toggleicon>
      <div class="p-1 bg-surface-muted">
        <SimpleIcon interactive :icon="collapsed ? IconCaretDownFilled : IconCaretUpFilled"
          class="text-content-muted" />
      </div>
    </template>
    <slot />
  </Panel>
</template>

<style scoped>
@import '@/assets/styles.css';

:deep(.p-panel-header),
:deep(.p-panel-content) {
  @apply !px-1 !py-2 ;
}

:deep(.p-panel-header) {
  @apply border-b mb-6;
}
</style>
