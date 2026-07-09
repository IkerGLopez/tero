### Jira

Manage issues and track project activity in Jira Cloud. The agent can work across the full issue lifecycle and project structure.

The agent can help with:

- **Issues** — create, read, update, and delete issues; add comments and attachments
- **Projects** — browse projects and retrieve project details and statuses
- **Search** — use JQL (Jira Query Language) to find issues across projects
- **Users** — look up user information

#### Authentication

This tool uses OAuth 2.0 (3LO). You need to create an OAuth app in the Atlassian developer console and provide its credentials.

**Steps to set up:**

1. Go to [developer.atlassian.com](https://developer.atlassian.com) and sign in.
2. Open **Developer console** and create a new app.
3. In the left menu, go to **Authorization** → click **Configure** next to **OAuth 2.0 (3LO)**.
4. Enter the **Callback URL** provided by your Tero instance and save.
5. Go to **Permissions** and add the **Jira API**. Enable the scopes you need:
   - `read:jira-work` — read issues, projects, and boards
   - `write:jira-work` — create and update issues
   - `read:jira-user` — read user information
6. Copy the **Client ID** and **Client secret** from the app settings.

Enter the Client ID, Client secret, and selected scopes in the tool configuration. After saving, each user will be prompted to authorize the connection with their Atlassian account.
