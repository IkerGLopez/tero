from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPMethod
import logging
from typing import Optional, cast

from httpx import HTTPStatusError
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import AgentToolConfig
from ..auth import (
    ToolAuthCallback,
    ToolAuthCallbackError,
    ToolAuthRepository,
    ToolAuthRequestException,
    ToolAuthToken,
    ToolAuthTokenType,
    ToolAuthTokenCallback,
    ToolAuthTokenRequest,
)
from ..core import load_schema
from ..openapi_tool import OpenApiTool


logger = logging.getLogger(__name__)
REDMINE_TOOL_ID = "redmine"


class RedmineTool(OpenApiTool):
    id: str = REDMINE_TOOL_ID
    name: str = "Redmine"
    description: str = "Manage issues, projects and log time"
    config_schema: dict = load_schema(__file__)
    _api_url: str = ""
    _api_key: Optional[str] = None

    def configure(self, agent, user_id: int, config: dict, db: AsyncSession, thread_id: Optional[int] = None):
        super().configure(agent, user_id, config, db, thread_id)
        self._api_url = config.get("url", "").rstrip("/")

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        self._api_key = self._get_secret("apiKey")
        prev_token = await ToolAuthRepository(self.db).find_token(self.user_id, self.agent.id, self.id)
        if (prev_config and prev_config.config != self.config 
            or self._api_key and prev_token and self._api_key != prev_token.access_token):
            await self.teardown()
        await self._check_auth()
        await self._save_api_key(self._api_key, self.db)

    async def teardown(self):
        await ToolAuthRepository(self.db).delete_token(self.user_id, self.agent.id, self.id)

    async def _check_auth(self):
        await self._invoke_rest_api(HTTPMethod.GET, f"{self._api_url}/users/current.json")

    async def _save_api_key(self, api_key: Optional[str], db: AsyncSession):
        if not api_key:
            return
        token = ToolAuthToken(
            user_id=self.user_id,
            agent_id=self.agent.id,
            tool_id=self.id,
            access_token=api_key,
            token_type=ToolAuthTokenType.BEARER,
            expires_in=None,
            scope=None,
            refresh_token=None,
            expires_at=None,
        )
        await ToolAuthRepository(db).save_token(token)

    @asynccontextmanager
    async def load(self) -> AsyncIterator["RedmineTool"]:
        try:
            await self._check_auth()
            yield self
        except HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
            raise

    async def auth(self, auth_callback: ToolAuthCallback):
        self._api_key = cast(ToolAuthTokenCallback, auth_callback).auth_token
        try:
            await self._check_auth()
        except HTTPStatusError as e:
            logger.warning(f"Invalid API key for agent {self.agent.id} tool {self.id} user {self.user_id}", exc_info=True)
            raise ToolAuthCallbackError("Invalid API key")
        await self._save_api_key(self._api_key, self.db)

    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        pass

    async def _add_auth_headers(self, headers: dict) -> dict:
        headers["X-Redmine-API-Key"] = await self._get_api_key()
        return headers

    async def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        token = await ToolAuthRepository(self.db).find_token(self.user_id, self.agent.id, self.id)
        if not token:
            raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
        return token.access_token
