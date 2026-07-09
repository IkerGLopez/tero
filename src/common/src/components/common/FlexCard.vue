<script lang="ts" setup>
import { ref, computed, getCurrentInstance } from 'vue';
const headerContent = ref<HTMLElement>()

// Checks if the header has a background image, if so, it will not have padding
const hasBgCover = () => {
  if (!headerContent.value) return false
  return headerContent.value.querySelector('.bg-cover') !== null
}

// important: default height for panel headers
withDefaults(defineProps<{
    headerHeight?: string;
    headerClass?: string;
}>(), {
    headerHeight: '74px'
});

const isInSidebar = computed(() => {
  const instance = getCurrentInstance();
  return instance?.parent?.type?.__name?.includes('Sidebar') || false;
});


</script>
<template>
    <div class="bg-surface h-full w-full flex flex-col rounded-2xl border shadow-sm" :class="{ 'px-4': !hasBgCover() }">
        <div class="flex flex-col h-full">
            <div
                class="border-b"
                :class="[{ 'py-4': !hasBgCover() }, { 'mb-4': !isInSidebar }, headerClass]"
                :style="{ height: headerHeight }"
                ref="headerContent">
                <slot name="header" />
            </div>
            <slot />
        </div>
    </div>
</template>
