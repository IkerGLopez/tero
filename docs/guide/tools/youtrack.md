### YouTrack

Report issues, log work time, and manage knowledge base articles in JetBrains YouTrack. The agent connects to the YouTrack MCP server built into your instance (requires YouTrack 2025.3 or later).

The agent can help with:

- **Issues** — search using YouTrack query language, create, update, assign, comment on, tag, and link issues
- **Time tracking** — log work time directly on issues
- **Knowledge Base** — search, read, create, and update articles
- **Projects** — find and inspect project configurations and custom field schemas
- **Users & groups** — look up users by name or email, list group members

#### Authentication

This tool uses a YouTrack permanent token and connects to the YouTrack MCP server built into your instance.

**Steps to set up:**

1. In YouTrack, click your **avatar** → **Profile** → **Account Security** tab.
2. In the **Tokens** section, click **New token**.
3. Enter a name and select the **YouTrack** scope, then click **Create token**.
4. Copy the token immediately — it is only shown once.
5. In the tool configuration, enter:
   - **Server URL**: your YouTrack MCP endpoint, e.g. `https://your-instance.youtrack.cloud/mcp`
   - **Token**: the permanent token you created
