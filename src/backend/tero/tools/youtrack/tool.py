from typing import cast

from ..core import load_schema
from ..mcp_tool import AuthType, BaseMcpTool


YOUTRACK_TOOL_ID = "youtrack"


class YouTrackTool(BaseMcpTool):
    id: str = YOUTRACK_TOOL_ID
    name: str = "YouTrack"
    description: str = "Report issues, manage agile boards and log work time"
    config_schema: dict = load_schema(__file__)

    def _get_auth_type(self) -> AuthType:
        return AuthType.BEARER_TOKEN

    def _get_mcp_server_url(self) -> str:
        server_url = cast(str, self.config.get("serverUrl") or "")
        if not server_url:
            raise ValueError("YouTrack MCP serverUrl is not configured")
        return server_url
