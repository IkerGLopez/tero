from ..core import load_schema
from ..mcp_tool import AuthType, BaseMcpTool


GITHUB_TOOL_ID = "github"
GITHUB_MCP_SERVER_URL = "https://api.githubcopilot.com/mcp/"


class GitHubTool(BaseMcpTool):
    id: str = GITHUB_TOOL_ID
    name: str = "GitHub"
    description: str = "Search code, manage repos, review PRs and track issues"
    config_schema: dict = load_schema(__file__)

    def _get_auth_type(self) -> AuthType:
        return AuthType.BEARER_TOKEN

    def _get_mcp_server_url(self) -> str:
        return GITHUB_MCP_SERVER_URL
