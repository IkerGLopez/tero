<script setup lang="ts">
import { ref } from 'vue';
import { useI18n } from 'vue-i18n'
import addAgentIcon from '@/assets/images/add-agent-icon.svg'

const { t } = useI18n()

const fileInput = ref<HTMLInputElement | null>(null);
const icon = defineModel<string|undefined>("icon")
const emit = defineEmits(["change"])

const onBrowseIcon = () => {
  fileInput.value?.click()
}

const onFileSelect = async (event: Event) => {
  await onIconSelect((event.target as HTMLInputElement).files!)
}

const onFileDrop = async (event: DragEvent) => {
  await onIconSelect(event.dataTransfer!.files)
}

const onIconSelect = async (files: FileList) => {
  const img = await loadImage(files[0])
  icon.value = await scaledImageBase64(img)
  emit("change")
}

const loadImage = (file: File): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      URL.revokeObjectURL(img.src);
      resolve(img)
    }
    img.onerror = error => reject(error)
    img.src = URL.createObjectURL(file);
  })
}

const scaledImageBase64 = (img: HTMLImageElement): string => {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!
  const width = 48
  const height = 48
  canvas.width = width
  canvas.height = height
  ctx.drawImage(img, 0, 0, width, height)
  const dataUrl = canvas.toDataURL('image/png')
  return dataUrl.split(',')[1]
}
</script>


<template>
  <div class="cursor-pointer relative border-1 rounded-full overflow-hidden bg-surface dark:bg-surface-muted">
    <input type="file" ref="fileInput" @change="onFileSelect" class="hidden" accept=".png" />
    <img @click="onBrowseIcon" @drop.prevent="onFileDrop" @dragover.prevent :src="icon ? `data:image/png;base64,${icon}` : addAgentIcon" class="w-16 h-16" v-tooltip.bottom="t('uploadIcon')"/>
  </div>
</template>

<i18n>
  {
    "en": {
      "generateIcon": "Generate icon",
      "uploadIcon": "Drag & drop or browse icon"
    },
    "es": {
      "generateIcon": "Generar icono",
      "uploadIcon": "Arrastra o selecciona un icono"
    }
  }
</i18n>