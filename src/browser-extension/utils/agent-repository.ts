import { browser } from 'wxt/browser';
import { Agent } from "./agent"

export const findAllAgents = async (): Promise<Agent[]> => {
  let { agents } = await browser.storage.local.get("agents")
  if (!agents) {
    agents = []
    await updateAgents(agents)
  }
  return agents.map((a: any) => Agent.fromJsonObject(a))
}

export const findAgentById = async (agentId: string): Promise<Agent | undefined> => {
  let agents = await findAllAgents()
  return agents.find(a => a.manifest.id === agentId)
}

export const addAgent = async (agent: Agent): Promise<void> => {
  let agents = await findAllAgents()
  agents.push(agent)
  agents.sort((a1, a2) => a1.manifest.id < a2.manifest.id ? -1 : 1)
  await updateAgents(agents)
}

export class ExistingAgentError extends Error {
}

export const updateAgents = async (agents: Agent[]) => {
  await browser.storage.local.set({ agents: agents })
}

export const removeAgent = async (agentId: string): Promise<void> => {
  let agents = await findAllAgents()
  agents = agents.filter(a => a.manifest.id !== agentId)
  await updateAgents(agents)
}

export const removeAllAgents = async (): Promise<void> => {
  await updateAgents([])
  await browser.storage.local.set({ prompts: {} })
}

export const findAgentsByHubUrl = async (hubUrl: string): Promise<Agent[]> => {
  const agents = await findAllAgents()
  return agents.filter(a => a.url === hubUrl)
}

export const removeAgentsByHubUrl = async (hubUrl: string): Promise<void> => {
  const agents = await findAllAgents()
  const filteredAgents = agents.filter(a => a.url !== hubUrl)
  await updateAgents(filteredAgents)
}
