### GitHub

Search code, manage repositories, review pull requests, and track issues — all through the official GitHub MCP server.

The agent can help with:

- **Issues** — create, read, update, comment on, and label issues
- **Pull requests** — create, review, merge, and comment on PRs; request reviews
- **Repositories** — browse files and commits, create branches, browse releases and tags
- **Users** — look up user profiles and organization members
- **Context** — access information about the authenticated user and their teams

::: info
Only the GitHub core tools are currently supported. Advanced toolsets such as Actions, Projects, Security, Discussions, and Notifications are not supported.
:::

#### Authentication

This tool uses a GitHub Personal Access Token. The token is used for all requests on behalf of the agent.

**Steps to get a token:**

1. Go to GitHub → **Settings** → **Developer settings** → **Personal access tokens**.
2. Create a new token (classic or fine-grained).
3. Grant the scopes your agent needs — for example `repo` for full repository access, `read:org` for organization data.
4. Copy the token and paste it into the tool configuration.

::: tip Fine-grained tokens
Fine-grained tokens let you limit access to specific repositories and permissions, which is a safer choice for shared agents.
:::
