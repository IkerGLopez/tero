<script lang="ts" setup>
import { ref, onBeforeMount, onBeforeUnmount, toRaw, computed } from 'vue'
import { browser } from 'wxt/browser'
import { useToast } from 'vue-toastification'
import { IconAlertCircleFilled } from '@tabler/icons-vue'
import { useI18n } from 'vue-i18n'
import { BrowserMessage, ActiveTabListener, ToggleSidebar, ActivateAgent, AgentActivation, InteractionSummary, ResizeSidebar } from '~/utils/browser-message'
import { Agent, AgentType } from '~/utils/agent'
import { TabState, TabDisplayMode, ChatMessage } from '~/utils/tab-state'
import { findTabState, saveTabState } from '~/utils/tab-state-repository'
import { findAgentSession } from '~/utils/agent-session-repository'
import { FlowStepError } from '~/utils/flow'
import { HttpServiceError } from '~/utils/http'
import { AuthService } from '~/utils/auth'
import ToastMessage from '~/components/ToastMessage.vue'
import CopilotChat from '~/components/CopilotChat.vue'
import CopilotList from '~/components/CopilotList.vue'
import 'highlight.js/styles/stackoverflow-light.css'

const toast = useToast()
const { t } = useI18n()

const agent = ref<Agent>()
let sidebarSize = 400
const minSidebarSize = 200
const minimizedSidebarSize = 120
let lastResizePos = 0
const messages = ref<ChatMessage[]>([])
const displayMode = ref(TabDisplayMode.CLOSED)

onBeforeMount(async () => {
  await restoreTabState()
  // we collect messages until we get all pending to avoid potential message order.
  // This may happen if listener is up, sends ActivateTabListener and while message
  // is being processed (and pending messages collected and returned) the service worker sends messages for processing here
  let collectedMessages: any[] | undefined = []
  browser.runtime.onMessage.addListener((m: any) => {
    if (collectedMessages === undefined) {
      onMessage(m)
    } else {
      collectedMessages.push(m)
    }
  })
  const pendingMessages: any = await sendToServiceWorker(new ActiveTabListener(true))
  if (pendingMessages) {
    for (const m of pendingMessages) {
      onMessage(m)
    }
  }
  for (const m of collectedMessages) {
    onMessage(m)
  }
  collectedMessages = undefined
  await resumeFlow()
})

const restoreTabState = async () => {
  const tabId = await getCurrentTabId()
  const tabState = await findTabState(tabId)
  if (tabState) {
    sidebarSize = tabState.sidebarSize
    displayMode.value = tabState.displayMode
    agent.value = tabState.agent
    messages.value = tabState.messages
  }
  if (displayMode.value !== TabDisplayMode.CLOSED) {
    resizeSidebar(sidebarSize)
  }
}

const sendToServiceWorker = async (msg: BrowserMessage): Promise<any> => {
  return await browser.runtime.sendMessage(msg)
}

const resumeFlow = async () => {
  try {
    const agentSession = await findAgentSession(await getCurrentTabId())
    await agentSession?.resumeFlow(onAgentResponse)
  } catch (error) {
    onAgentError(error)
  }
}

onBeforeUnmount(async () => {
  const tabId = await getCurrentTabId()
  await saveTabState(tabId, new TabState(sidebarSize, displayMode.value, toRaw(messages.value), toRaw(agent.value)))
  // wait for message processing so we are sure that before loading new content this has been marked as not ready for messages
  await sendToServiceWorker(new ActiveTabListener(false))
})

const onMessage = (m: any) => {
  const msg = BrowserMessage.fromJsonObject(m)
  if (msg instanceof ToggleSidebar) {
    onToggleSidebar()
  } else if (msg instanceof AgentActivation) {
    onAgentActivation(msg)
  } else if (msg instanceof InteractionSummary) {
    onInteractionSummary(msg)
  }
}

const onToggleSidebar = () => {
  displayMode.value = displayMode.value === TabDisplayMode.CLOSED ? TabDisplayMode.FULL : TabDisplayMode.CLOSED
  resizeSidebar(displayMode.value === TabDisplayMode.FULL ? sidebarSize : 0)
}

const resizeSidebar = async (size: number, height?: string, position?: "top" | "bottom") => {
  browser.tabs.sendMessage(await getCurrentTabId(), new ResizeSidebar(size, height, position))
}

const getCurrentTabId = async (): Promise<number> => {
  const ret = await browser.tabs.getCurrent()
  return ret.id!
}

const onStartResize = (e: MouseEvent) => {
  lastResizePos = e.screenX
  window.document.body.className = 'resizing'
  window.addEventListener('mousemove', onResize)
  window.addEventListener('mouseup', onEndResize)
}

const onResize = async (e: MouseEvent) => {
  e.preventDefault()
  const delta = lastResizePos - e.screenX
  lastResizePos = e.screenX
  sidebarSize += delta
  if (sidebarSize < minSidebarSize) {
    sidebarSize = minSidebarSize
  }
  resizeSidebar(sidebarSize)
}

const onEndResize = () => {
  window.document.body.className = ''
  window.removeEventListener('mousemove', onResize)
  window.removeEventListener('mouseup', onEndResize)
}

const onCloseSidebar = async () => {
  if (agent.value) {
    const tabId = await getCurrentTabId()
    const session = await findAgentSession(tabId)
    if (session) {
      await removeAgentSession(tabId)
      try {
        await session.close()
      } catch (error) {
        console.error("Problem closing session", error)
      }
    }
    agent.value = undefined
    messages.value = []
  } else {
    displayMode.value = TabDisplayMode.CLOSED
    resizeSidebar(0)
  }
}

const onMinimizeSidebar = async () => {
  displayMode.value = TabDisplayMode.MINIMIZED
  resizeSidebar(minimizedSidebarSize, "100px", "bottom")
}

const onRestoreSidebar = async () => {
  if (!isMinimized.value) {
    return
  }
  resizeSidebar(sidebarSize, "100%", "top")
  // this delay avoids a noticeable flash with the full sidebar content but with minimized size and position
  setTimeout(() => {
    displayMode.value = TabDisplayMode.FULL
  }, 100)
}

const isMinimized = computed(() => {
  return displayMode.value === TabDisplayMode.MINIMIZED
})

const onActivateAgent = async (agentId: string) => {
  const tab = await browser.tabs.getCurrent()
  sendToServiceWorker(new ActivateAgent(agentId, tab.url!))
}

const onAgentActivation = async (msg: AgentActivation) => {
  if (displayMode.value === TabDisplayMode.CLOSED) {
    onToggleSidebar()
  }
  if (!msg.success) {
    if (msg.errorStatus === 401 && msg.agent.manifest.auth) {
      const authService = new AuthService(msg.agent.manifest.auth)
      await authService.ensureAuthenticated()
      await onActivateAgent(msg.agent.manifest.id!)
      return
    }

    const text = t('activationError', { agentName: msg.agent.manifest.name, contactEmail: msg.agent.manifest.contactEmail })
    toast.error({ component: ToastMessage, props: { message: text } }, { icon: IconAlertCircleFilled })
  } else {
    const activatedAgent = Agent.fromJsonObject(msg.agent)
    agent.value = activatedAgent

    const welcomeMessage = activatedAgent.manifest.welcomeMessage
    if (activatedAgent.type === AgentType.StandaloneAgent && welcomeMessage) {
      messages.value.push(ChatMessage.agentMessage(welcomeMessage))
    }
  }
}

const onInteractionSummary = (msg: InteractionSummary) => {
  const text = msg.text ? msg.text : t('interactionSummaryError', { contactEmail: agent.value!.manifest.contactEmail })
  const messagePosition = getLastMessage().isComplete ? messages.value.length : messages.value.length - 1
  messages.value.splice(messagePosition, 0, msg.success ? ChatMessage.agentMessage(text) : ChatMessage.agentErrorMessage(text))
}

const getLastMessage = () => {
  return messages.value[messages.value.length - 1]
}

const onUserMessage = async (text: string) => {
  try {
    const lastMessage = getLastMessage()
    messages.value.push(ChatMessage.userMessage(text))
    messages.value.push(ChatMessage.agentMessage())
    const agentSession = await findAgentSession(await getCurrentTabId())
    await agentSession!.processUserMessage(text, onAgentResponse, lastMessage?.id)
  } catch (error) {
    onAgentError(error)
  }
}

const onUserStopResponse = async () => {
  try {
    const agentSession = await findAgentSession(await getCurrentTabId())
    await agentSession!.stopResponse()
    getLastMessage().isComplete = true
  } catch (error) {
    onAgentError(error)
  }
}

const audioTranscriber = async (blob: Blob) : Promise<string> => {
  try {
    const agentSession = await findAgentSession(await getCurrentTabId())
    return await agentSession!.transcribeAudio(blob)
  } catch (error) {
    onAgentError(error)
    return ""
  }
}

const onAgentResponse = (part: MessagePart) => {
  const lastMessage = getLastMessage()
  if (part.message !== undefined) {
    lastMessage.isComplete = part.complete ?? false
    lastMessage.text += part.message
  } else if (part.metadata !== undefined) {
    lastMessage.id = part.metadata.answerMessageId
  }
}

const onAgentError = (error: any) => {
  const defaultErrorMessage = t('agentAnswerError', { contactEmail: agent.value!.manifest.contactEmail })
  // exceptions from http methods are already logged so no need to handle them
  if (!(error instanceof HttpServiceError)) {
    console.warn("Problem processing agent answer", error)
  }
  let text = null
  if (error instanceof HttpServiceError && error.detail) {
    text = typeof error.detail === 'string' ? error.detail : defaultErrorMessage
  } else if (error instanceof FlowStepError && error.errorCode === 'MissingElement') {
    text = t("flowStep" + error.errorCode, { selector: error.step.selector, contactEmail: agent.value!.manifest.contactEmail })
  } else {
    text = defaultErrorMessage
  }
  const lastMessage = getLastMessage()
  lastMessage.isComplete = true
  if (!lastMessage.text) {
    lastMessage.text += text
    lastMessage.isSuccess = false
  } else {
    messages.value.push(ChatMessage.agentErrorMessage(text))
  }
}

const onNewChat = async () => {
  const session = await findAgentSession(await getCurrentTabId())
  if (session) {
    try {
      await session.close()
    } catch (error) {
      console.error("Problem closing session", error)
    }
  }
  messages.value = []
  await onActivateAgent(agent.value!.manifest.id!)
}

const sidebarClasses = computed(() => [
  'fixed flex flex-col bg-surface border',
  isMinimized.value
    ? 'bottom-4 right-4 rounded-full shadow-lg cursor-pointer hover:shadow-xl transition-shadow'
    : 'm-2 -left-2 w-full h-[calc(100%-16px)] rounded-tl-3xl rounded-bl-3xl'
])
</script>

<template>
  <div :class="sidebarClasses" id="sidebar" @click="onRestoreSidebar">
    <div v-if="!isMinimized" class="absolute left-0 z-auto cursor-ew-resize w-2 h-full" @mousedown="onStartResize"/>
    <CopilotChat v-if="agent" :messages="messages" :agent="agent" :audio-transcriber="audioTranscriber"
      :error-handler="onAgentError" :minimized="isMinimized" @user-message="onUserMessage" @stop-response="onUserStopResponse"
      @close="onCloseSidebar" @minimize="onMinimizeSidebar" @new-chat="onNewChat" />
    <CopilotList v-if="!agent" @activate-agent="onActivateAgent" @close="onCloseSidebar" />
  </div>
</template>

<style scoped>
#sidebar ::-webkit-scrollbar {
  width: 7px;
  background-color: var(--color-auxiliar-gray);
  border-radius: 3px;
}

#sidebar ::-webkit-scrollbar-thumb {
  background-color: var(--color-primary);
  border-radius: 3px;
}

#sidebar ::-webkit-scrollbar-thumb:hover {
  background-color: var(--color-primary);
}
</style>

<i18n lang="json">
{
  "en": {
    "activationError": "Could not activate {agentName}. You can try again and if the issue persists then contact [{agentName} support](mailto:{contactEmail}?subject=Activation%20issue)",
    "interactionSummaryError": "I could not process some information from the current site. This might impact the information and answers I provide. If the issue persists please contact [support](mailto:{contactEmail}?subject=Interaction%20issue)",
    "agentAnswerError": "I can't help with that message. Edit it or send a new one. If the problem continues, [contact the support team](mailto:{contactEmail}?subject=Question%20issue)",
    "flowStepMissingElement": "I could not find the element '{selector}'. This might be due to recent changes in the page which I am not aware of. Please try again and if the issue persists contact [support](mailto:{contactEmail}?subject=Navigation%20element)."
  },
  "es": {
    "activationError": "No se pudo activar {agentName}. Puedes intentar de nuevo y si el problema persiste contactar al [soporte de {agentName}](mailto:{contactEmail}?subject=Activation%20issue)",
    "interactionSummaryError": "No pude procesar informacion generada por la página actual. Esto puede impactar en la información y respuestas que te puedo dar. Si el problema persiste por favor contacta a [soporte](mailto:{contactEmail})?subject=Interaction%20issue",
    "agentAnswerError": "No puedo ayudarte con ese mensaje. Probá editarlo o enviar uno nuevo. Si el problema continúa, podés [contactar al equipo de soporte](mailto:{contactEmail}?subject=Question%20issue)",
    "flowStepMissingElement": "No pude encontrar el elemento '{selector}'. Esto puede ser debido a cambios recientes en la página de los cuales no tengo conocimiento. Por favor intenta de nuevo y si el problema persiste contacta a [soporte](mailto:{contactEmail}?subject=Navigation%20element)."
  }
}
</i18n>
