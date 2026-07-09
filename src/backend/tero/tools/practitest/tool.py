from typing import cast

from ..core import load_schema
from ..mcp_tool import AuthType, BaseMcpTool


PRACTITEST_TOOL_ID = "practitest"


class PractiTestTool(BaseMcpTool):
    id: str = PRACTITEST_TOOL_ID
    name: str = "PractiTest"
    description: str = "Manage tests, test sets and requirements"
    config_schema: dict = load_schema(__file__)

    def _get_auth_type(self) -> AuthType:
        return AuthType.BEARER_TOKEN

    def _get_mcp_server_url(self) -> str:
        server_url = cast(str, self.config.get("serverUrl") or "")
        if not server_url:
            raise ValueError("PractiTest MCP serverUrl is not configured")
        return server_url
