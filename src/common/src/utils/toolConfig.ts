import { IconBrain, IconQuestionMark, IconWorld, IconDeviceDesktopBolt, type Icon } from '@tabler/icons-vue'
import { h, type SVGAttributes } from 'vue'

import mcpIcon from '../assets/images/mcp-icon.svg'
import jiraIcon from '../assets/images/jira-icon.svg'
import redmineIcon from '../assets/images/redmine-icon.svg'
import githubIcon from '../assets/images/github-icon.svg'
import youtrackIcon from '../assets/images/youtrack-icon.svg'
import practitestIcon from '../assets/images/practitest-tool.svg'

const iconFromImage = (imageSrc: string): Icon => {
  return (props: SVGAttributes) => {
    const { class: className, ...restProps } = (props ?? {}) as SVGAttributes & { class?: unknown }

    return h('img', {
      src: imageSrc,
      width: 20,
      height: 20,
      class: ['tool-menu-icon', className],
      ...restProps
    })
  }
}

const toolIcons: Record<string, Icon> = {
  docs: IconBrain,
  web: IconWorld,
  'mcp-*': iconFromImage(mcpIcon),
  jira: iconFromImage(jiraIcon),
  github: iconFromImage(githubIcon),
  redmine: iconFromImage(redmineIcon),
  youtrack: iconFromImage(youtrackIcon),
  practitest: iconFromImage(practitestIcon),
  browser: IconDeviceDesktopBolt
}

export const defaultToolNames: Record<string, string> = {
  docs: 'Docs',
  web: 'Web',
  mcp: 'MCP',
  jira: 'Jira',
  github: 'GitHub',
  youtrack: 'YouTrack',
  practitest: 'PractiTest',
  redmine: 'Redmine',
  browser: 'Browser'
}

export const findToolIcon = (toolId: string): Icon => {
  const key = toolId.includes('-') ? toolId.split('-', 1)[0] + '-*' : toolId
  return toolIcons[key] || IconQuestionMark
}

export const buildToolConfigName = (toolId: string): string => {
  const [prefix, suffix] = toolId.split('-', 2)
  return suffix && suffix !== '*' ? suffix : defaultToolNames[prefix] || prefix
}
