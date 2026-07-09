### Redmine

Manage issues, projects, and time tracking in Redmine.

::: warning Enterprise Edition
This tool is only available in the Tero Enterprise Edition.
:::

The agent can help with:

- **Issues** — create, read, update, and delete issues; add notes; change status and assignee
- **Time tracking** — log and review time entries on issues and projects
- **Projects** — browse projects and versions
- **Search** — find issues across projects using filters

#### Authentication

This tool uses a Redmine API key for authentication.

::: tip Prerequisite
A Redmine administrator must first enable the REST API under **Administration → Settings → API → Enable REST API**.
:::

**Steps to set up:**

1. In Redmine, click your account name in the top-right → **My account**.
2. In the right-hand pane, find your **API access key** (generate one if it doesn't exist yet).
3. In the tool configuration, enter:
   - **URL**: your Redmine instance URL, e.g. `https://redmine.example.com`
   - **API Key**: the key from your Redmine account
