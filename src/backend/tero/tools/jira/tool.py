from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPMethod
import logging
from typing import Any, Optional, cast

from pydantic import AnyHttpUrl
from sqlmodel import Field, SQLModel, and_, select, delete
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import Agent, AgentToolConfig
from ...core.repos import scalar
from ..core import load_schema
from ..openapi_tool import OpenApiTool
from ..auth import AgentToolOauth, ToolAuthCallback, ToolOAuthCallback, ToolOAuthClientInfo, ToolOAuthClientInfoRepository, ToolAuthRepository, OAuthMetadata, ToolAuthCallbackError, ToolAuthRequestException, ToolAuthTokenRequest


logger = logging.getLogger(__name__)
JIRA_TOOL_ID = "jira"
SWAGGER_URL = "https://developer.atlassian.com/cloud/jira/platform/swagger-v3.v3.json"
JIRA_BASE_API_URL = "https://api.atlassian.com"


class JiraToolConfig(SQLModel, table=True):
    __tablename__ : Any = "jira_tool_config"
    agent_id: int = Field(primary_key=True)
    cloud_id: str


class JiraToolConfigRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_by_agent_id(self, agent_id: int) -> Optional[JiraToolConfig]:
        stmt = select(JiraToolConfig).where(JiraToolConfig.agent_id == agent_id)
        result = await self._db.exec(stmt)
        return result.first()
    
    async def save(self, config: JiraToolConfig):
        self._db.add(config)
        await self._db.commit()

    async def delete(self, agent_id: int):
        stmt = scalar(delete(JiraToolConfig).where(and_(JiraToolConfig.agent_id == agent_id)))
        await self._db.exec(stmt)
        await self._db.commit()


class JiraTool(OpenApiTool):
    id: str = JIRA_TOOL_ID
    name: str = "Jira"
    description: str = "Manage issues and track project activity"
    config_schema: dict = load_schema(__file__)
    _client_secret: Optional[str] = None

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        client_info_repo = ToolOAuthClientInfoRepository(self.db)
        prev_client_info = await client_info_repo.find_by_ids(self.agent.id, self.id)
        client_secret = self._get_secret("clientSecret")
        client_id = self.config["clientId"]
        if prev_config and prev_config.config != self.config or prev_client_info and client_secret and prev_client_info.client_secret != client_secret:
            if not client_secret and prev_client_info and prev_client_info.client_id == client_id:
                client_secret = prev_client_info.client_secret
            await self.teardown()
        if client_secret:
            await client_info_repo.save(ToolOAuthClientInfo(
                agent_id=self.agent.id,
                tool_id=self.id,
                client_id=client_id,
                client_secret=client_secret,
                token_endpoint_auth_method="client_secret_post",
                scope=" ".join(self.config["scope"])))
        async with self.load():
            pass
        
    @asynccontextmanager
    async def load(self) -> AsyncIterator['JiraTool']:
       self._oauth = await self._load_oauth()
       await self._oauth.solve_tokens()
       cloud_id = await self._find_cloud_id()
       self._api_url = f"{JIRA_BASE_API_URL}/ex/jira/{cloud_id}"
       yield self

    async def _load_oauth(self) -> AgentToolOauth:
        base_url = "https://auth.atlassian.com"
        oauth_metadata = OAuthMetadata(
            issuer=AnyHttpUrl(base_url),
            authorization_endpoint=AnyHttpUrl(f"{base_url}/authorize"),
            token_endpoint=AnyHttpUrl(f"{base_url}/oauth/token")
        )
        client_info = await ToolOAuthClientInfoRepository(self.db).find_by_ids(self.agent.id, self.id)
        if not client_info or not client_info.scope:
            raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
        # add offline_access scope to be able to refresh tokens
        return AgentToolOauth(base_url, oauth_metadata, cast(str, cast(ToolOAuthClientInfo, client_info).scope) + " offline_access", self.agent.id, self.id, self.user_id, self.db)

    async def _find_cloud_id(self):
        repo = JiraToolConfigRepository(self.db)
        jira_config = await repo.find_by_agent_id(self.agent.id)
        if jira_config:
            return jira_config.cloud_id
        resp = await self._invoke_rest_api(HTTPMethod.GET, f"{JIRA_BASE_API_URL}/oauth/token/accessible-resources")
        ret = next(resource["id"] for resource in resp)
        await repo.save(JiraToolConfig(agent_id=self.agent.id, cloud_id=ret))
        return ret
    
    async def _add_auth_headers(self, headers: dict) -> dict:
        tokens = await cast(AgentToolOauth, self._oauth).solve_tokens()
        if tokens:
            headers["Authorization"] = f"Bearer {tokens.access_token}"
        return headers

    async def auth(self, auth_callback: ToolAuthCallback):
        state = await ToolAuthRepository(self.db).find_state(self.user_id, self.id, cast(ToolOAuthCallback, auth_callback).state)
        if not state:
            raise ToolAuthCallbackError("OAuth state not found")
        oauth = await self._load_oauth()
        await oauth.callback(cast(ToolOAuthCallback, auth_callback), state)

    async def teardown(self):
        await ToolAuthRepository(self.db).delete_token(self.user_id, self.agent.id, self.id)
        await ToolOAuthClientInfoRepository(self.db).delete(self.agent.id, self.id)
        await JiraToolConfigRepository(self.db).delete(self.agent.id)

    async def _load_api_spec(self) -> dict:
        ret = await super()._load_api_spec()
        schemas = ret["components"]["schemas"]
        # using simplified schema instead of the original one from https://unpkg.com/@atlaskit/adf-schema@49.0.1/dist/json-schema/v1/full.json 
        # since original schema is huge, consuming time, tokens and making llm confused with so much information
        # additionally, just having version after content in doc_node makes the llm to generate a call without the version attribute, which makes the request to fail
        doc_node_schema = await self._load_json("simplified-doc-node-schema.json")
        schemas.update(doc_node_schema["definitions"])
        return ret

    # there is a limitation of up to 128 functions that can be passed to OpenAI, and JIRA API has more than 590 methods. This method filters the most common and used ones.
    def _should_include_operation(self, path: str, method: str) -> bool:
        base_path = "/rest/api/3"
        issues_path = f"{base_path}/issue"
        issue_path = f"{issues_path}/{{issueIdOrKey}}"
        comments_path = f"{issue_path}/comment"
        properties_path = f"{issue_path}/properties"
        search_path = f"{base_path}/search"
        projects_path = f"{base_path}/project"
        project_path = f"{projects_path}/{{projectIdOrKey}}"
        return path in [
            issues_path, issue_path, f"{issue_path}/assignee", f"{issue_path}/attachments", f"{issue_path}/changelog", 
            comments_path, f"{comments_path}/{{id}}", properties_path, f"{properties_path}/{{propertyKey}}", f"{issue_path}/transitions", 
            f"{search_path}/approximate-count", f"{search_path}/jql", f"{projects_path}/search",
            f"{base_path}/myself", f"{base_path}/users/search"] \
                or (method == "get" and path in [f"{project_path}", f"{project_path}/statuses"])
    
    def _handle_schema_without_type(self, schema: dict, schemas: dict, refs: set) -> None:
        # Fix Jira schema which does not properly define the schema for comments.
        if "Atlassian Document Format" in schema.get("description", ""):
            self._refactor_ref(schema, "doc_node", schemas, refs)
    
    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        pass
