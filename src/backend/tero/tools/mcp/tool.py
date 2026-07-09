import logging
from typing import Optional, cast
from pydantic import AnyHttpUrl
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import Agent
from ..core import load_schema
from ..mcp_tool import AuthType, BaseMcpTool


MCP_TOOL_ID = "mcp"
logger = logging.getLogger(__name__)


class McpTool(BaseMcpTool):
    id: str = MCP_TOOL_ID + "-*"
    name: str = "MCP"
    description: str = "Use tools from any MCP server"
    config_schema: dict = load_schema(__file__)
    _bearer_secret_key: str = "bearerToken"

    def configure(self, agent: Agent, user_id: int, config: dict, db: AsyncSession, thread_id: Optional[int] = None):
        super().configure(agent, user_id, config, db, thread_id)
        server_url = cast(str, config.get("serverUrl", ""))
        self.id = MCP_TOOL_ID + "-" + cast(str, AnyHttpUrl(server_url).host)

    def _get_auth_type(self) -> AuthType:
        return AuthType.from_string(self.config.get("authType", AuthType.OAUTH.value))

    def _get_mcp_server_url(self) -> str:
        return cast(str, self.config.get("serverUrl", ""))

    def _get_transport(self) -> str:
        url = self._get_mcp_server_url()
        return "sse" if url.endswith("/sse") else "streamable_http"

    def _get_connection_id(self) -> str:
        return cast(str, AnyHttpUrl(self._get_mcp_server_url()).host)

