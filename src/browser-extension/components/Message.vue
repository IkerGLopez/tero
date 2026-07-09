<script lang="ts" setup>
import { computed, nextTick, ref, onMounted, onBeforeUnmount } from 'vue'
import { useI18n } from 'vue-i18n'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/base16/gigavolt.min.css'
import MarkdownItPlantuml from 'markdown-it-plantuml'
import { IconExclamationCircle, IconCircleFilled, IconCirclePlus } from '@tabler/icons-vue'
import * as echarts from 'echarts'
import moment from 'moment'
import { Agent } from '~/utils/agent'

const props = defineProps<{ text: string, isUser: boolean, isComplete: boolean, isSuccess: boolean, agent: Agent }>()
const emit = defineEmits<{
  (e: 'promptCreate', text: string): void
}>()

const { t } = useI18n()
const renderedMsg = computed(() => props.isUser ? props.text.replaceAll('\n', '<br/>') : renderMarkDown(props.text))
const messageElement = ref<HTMLElement | null>(null);
const resizeObserver: ResizeObserver = new ResizeObserver(onResize)
var chart: any;
var prevWidth: number = 0;

function renderMarkDown(text: string) {
  let md = new MarkdownIt({
    highlight: (code: string, lang: string) => {
      let ret = code
      if (lang && hljs.getLanguage(lang)) {
        try {
          ret = hljs.highlight(code, { language: lang }).value
        } catch (__) { }
      }
      return '<pre><code class="hljs">' + ret + '</code></pre>'
    }
  })
  useTargetBlankLinks(md)
  useEcharts(md)
  md.use(MarkdownItPlantuml)
  return md.render(text)
}

function useTargetBlankLinks(md: MarkdownIt) {
  let defaultRender = md.renderer.rules.link_open || function (tokens, idx, options, env, self) {
    return self.renderToken(tokens, idx, options)
  }
  md.renderer.rules.link_open = function (tokens, idx, options, env, self) {
    tokens[idx].attrSet('target', '_blank')
    return defaultRender(tokens, idx, options, env, self)
  }
}

function useEcharts(md: MarkdownIt) {
  const defaultRender = md.renderer.rules.fence || function (tokens, idx, options, env, self) {
    return self.renderToken(tokens, idx, options);
  };
  md.renderer.rules.fence = function (tokens, idx, options, env, self) {
    const token = tokens[idx];
    const code = token.content.trim();
    if (token.info === 'echarts') {
      nextTick().then(() => {
        const container = messageElement.value!;
        const chartDiv = container.querySelector('.echarts');
        const chartData = container.querySelector('.echarts-data');
        if (chartDiv && chartData) {
          chart = echarts.init(chartDiv as HTMLDivElement);
          const options = JSON.parse(chartData.textContent || '');
          solveEchartsFormatter(options.xAxis.axisLabel);
          solveEchartsFormatter(options.xAxis.axisPointer.label);
          chart.setOption(options);
        }
      });
      return `<div class="echarts" style="width: 100%; height: 200px;"></div><div class="echarts-data" style='display:none'>${code}</div>`;
    }
    return defaultRender(tokens, idx, options, env, self);
  };
}

function solveEchartsFormatter(obj: any) {
  if (obj && obj.formatter) {
    if (obj.formatter.name === 'formatEpoch') {
      obj.formatter = formatEpoch(obj.formatter)
    }
  }
}

function formatEpoch(config: any): (value: any) => string {
  return (value: any) => {
    if (typeof value === 'object') {
      value = value.value;
    }
    const time = moment(parseInt(value))
    return time.format(config.format);
  }
}

function onResize() {
  if (messageElement.value!.scrollWidth != prevWidth && chart) {
    prevWidth = messageElement.value!.scrollWidth;
    chart.resize();
  }
}

onMounted(() => {
  if (messageElement.value) {
    resizeObserver.observe(messageElement.value)
  }
})

onBeforeUnmount(() => {
  if (messageElement.value) {
    resizeObserver.unobserve(messageElement.value)
  }
})
</script>

<template>
  <div class="flex flex-col mb-1 p-1 min-w-7" :class="!isSuccess ? ['border-red-500', 'border-b'] : []">
    <div class="flex flex-row items-start">
      <div class="flex w-5 h-5 mr-1 items-center flex-shrink-0">
        <template v-if="isUser" class="">
          <IconCircleFilled class="text-primary" />
        </template>
        <template v-else-if="!isUser && isSuccess">
          <img :src="agent.logo" class="rounded-full object-cover" />
        </template>
        <template v-else>
          <IconExclamationCircle class="text-error" />
        </template>
      </div>
      <div class="flex flex-col">
        <template v-if="text">
          <div v-html="renderedMsg" ref="messageElement"
            class="flex flex-col text-sm font-light leading-tight gap-2 rendered-msg" />
        </template>
        <div class="ml-3 dot-pulse" v-if="!isComplete" />
      </div>
      <div class="flex-auto flex justify-end ml-1">
        <CopyButton v-if="!isUser && text" :text="text" :html="renderedMsg" />
        <SimpleIcon interactive v-if="isUser && text" @click="emit('promptCreate', text)" :icon="IconCirclePlus"/>
      </div>
    </div>
  </div>
</template>
<style lang="scss">
@use 'three-dots' with ($dot-width: 5px,
  $dot-height: 5px,
  $dot-color: var(--color-primary));

.rendered-msg pre {
  padding: 15px;
  background: var(--color-surface-muted);
  border-radius: 8px;
  text-wrap: wrap;
}

// Fix: Inadequate gap between code blocks within list items.
.rendered-msg li pre {
  margin-bottom: 10px;
}

.rendered-msg pre {
  box-shadow: var(--shadow);
}

.rendered-msg pre code.hljs {
  padding: 0px;
}

div a {
  color: var(--color-primary);
  text-decoration: none;
}

.rendered-msg table {
  width: 100%;
  box-shadow: var(--shadow);
}

.rendered-msg thead tr {
  background-color: var(--color-surface-muted);
}

.rendered-msg th,
.rendered-msg td {
  padding: var(--half-spacing);
  border: var(--border);
}

.rendered-msg tbody tr:hover {
  background-color: var(--color-auxiliar-gray);
}

.echarts {
  box-shadow: var(--shadow);
  border-radius: var(--spacing);
  width: 100%;
  padding: var(--half-spacing);
}

.rendered-msg>img {
  box-shadow: var(--shadow);
  border-radius: var(--spacing);
  width: fit-content;
}
</style>

<i18n>
{
  "en": {
    "you": "You"
  },
  "es": {
    "you": "Tú"
  }
}
</i18n>
