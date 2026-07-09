### MCP

Connect to any [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server and expose its tools to the agent. This lets you integrate any service that provides an MCP interface without needing a dedicated Tero tool.

Once connected, the agent automatically discovers and can use all tools exposed by the MCP server — whatever those tools do (querying APIs, managing resources, reading data) becomes available in the agent's context.

#### Authentication

Two authentication modes are supported:

**OAuth** — the agent creator provides the server URL and each user authorizes via the OAuth flow when they first use the tool.

**Bearer Token** — the agent creator provides the server URL and a static bearer token used for all requests.

Some servers also require custom HTTP headers (for example, a workspace ID or API version header). Use the **Custom headers** field to add any additional headers the server needs.

::: tip Transport
The tool supports both SSE (`/sse` endpoint) and streamable HTTP transports. The correct transport is selected automatically based on the server URL.
:::
