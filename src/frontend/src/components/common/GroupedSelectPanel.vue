<script lang="ts" setup>
import { ref, toRef } from 'vue'
import { useElementSize } from '@vueuse/core'
import { Popover } from 'primevue'
import { useI18n } from 'vue-i18n'

export type GroupedSelectPanelOptionItem = {
  id: string
  name: string
  description: string
  costMultiplier?: number | null
}

export type GroupedSelectPanelOptionGroup = {
  id: string
  name: string
  description?: string
  children?: GroupedSelectPanelOptionItem[]
}

const { t } = useI18n()

const props = defineProps<{
  searchPlaceholder: string
  container?: HTMLElement
  anchor?: HTMLElement
  showLoadMore: boolean
  height?: string
}>()

const emit = defineEmits<{
  (e: 'search', value: string): void
  (e: 'loadMore'): void
}>()

const searchQuery = ref('')
const popoverRef = ref<InstanceType<typeof Popover>>()

const { width: popoverWidth } = useElementSize(toRef(props, 'container'))

const onSearch = (value: string | number | undefined) => {
  const next = String(value ?? '')
  searchQuery.value = next
  emit('search', next)
}

const onShowDropdown = () => {
  popoverRef.value?.toggle({
    currentTarget: props.anchor ?? props.container
  } as unknown as Event)
}

defineExpose({ onShowDropdown })
</script>

<template>
  <Popover ref="popoverRef" class="border rounded-2xl! pb-4 overflow-hidden!" :style="{ width: `${popoverWidth}px` }">
    <div class="relative flex flex-col gap-2">
      <div class="flex flex-col gap-2 w-full justify-between">
        <div class="flex flex-row gap-4 items-center w-full sticky top-0 z-10 border-b py-2 px-4">
          <InteractiveInput autofocus :model-value="searchQuery" @update:model-value="onSearch" :placeholder="searchPlaceholder" start-icon="IconSearch" class="flex-1 text-sm" />
        </div>
        <div
          class="flex flex-col w-full overflow-y-auto gap-2 px-4 pr-2"
          :style="{ maxHeight: '40vh', height: height ?? '30vh' }"
        >
          <slot name="content" />
          <div v-if="showLoadMore" class="flex items-center justify-center pb-2">
            <SimpleButton shape="square" @click="emit('loadMore')">{{ t('loadMore') }}</SimpleButton>
          </div>
        </div>
      </div>
    </div>
  </Popover>
</template>

<style>
.p-popover-content {
  padding: 0 !important;
}
</style>

<i18n lang="json">
{
  "en": {
    "loadMore": "Load more"
  },
  "es": {
    "loadMore": "Cargar más"
  }
}
</i18n>
