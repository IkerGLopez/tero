### Browser

Automate real browser interactions using Playwright. Unlike the Web tool which only reads static content, the Browser tool can interact with live web pages — useful for apps that require login, dynamic content, or multi-step navigation.

The agent can:

- **Navigate** — open URLs, follow links, handle redirects and page transitions
- **Interact** — click buttons, fill forms, select dropdowns, check checkboxes, upload files
- **Read** — extract text, read table data, capture the current state of dynamic pages
- **Screenshot** — take screenshots to capture the current state of a page
- **Wait** — wait for elements to appear or conditions to be met before acting

This tool requires no configuration by the agent creator.

::: warning Concurrent browser sessions
Tero currently only supports 1 active browser session at a time. If multiple users try to use it at the same time they may get a notification about this.
:::
