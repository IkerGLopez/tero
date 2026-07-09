### Web

Search the web and extract text content from public URLs. The agent can look up current information beyond its training data or read specific pages as part of answering a request.

The agent can:

- **Search** — query the web and retrieve a ranked list of relevant results with snippets
- **Extract** — read the full text content of public URLs, including documentation sites, blog posts, changelogs, and issue trackers

This tool requires no configuration by the agent creator. It is available when a Tero admin has configured a web search API key (Tavily or Google Custom Search) for the instance.

::: tip Search vs Extract
The agent uses **search** to find relevant pages for a query, and **extract** to read the full content of a specific URL. You can ask the agent to visit a URL directly, or let it search and pick the best result.
:::
