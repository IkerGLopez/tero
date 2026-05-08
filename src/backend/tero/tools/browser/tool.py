import aiofiles
import aiofiles.os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import re
from typing import List, Optional, Any, cast, Annotated

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, InjectedToolCallId
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from sqlmodel.ext.asyncio.session import AsyncSession
from json_schema_to_pydantic import create_model
from pydantic import BaseModel

from ...core.env import env
from ...files.domain import File, FileStatus, FileMetadata
from ...files.repos import FileRepository
from ..core import AgentTool, AgentToolConfig, load_schema, StatusUpdateCallbackHandler


logger = logging.getLogger(__name__)
BROWSER_TOOL_ID = "browser"


class ScreenshotPersistingTool(BaseTool):
    _PLAYWRIGHT_OUTPUT_DIR = "/tmp/playwright-output"

    def __init__(self, tool: BaseTool, user_id: int, thread_id: Optional[int], db: AsyncSession):
        super().__init__(name=tool.name, description=tool.description + ". IT IS IMPORTANT TO AVOID USING PROVIDED PATH TO GENERATE RESPONSES, unless the user explicitly asks for it, since the agent already provides other means to download the screenshot. IT IS VERY IMPORTANT TO NOT USE RETURNED URL AS SOURCE OF AN IMAGE.", args_schema=self._build_args_schema(cast(dict, tool.args_schema)), metadata=tool.metadata)
        self._mcp_tool = tool
        self._user_id = user_id
        self._db = db
        self._thread_id = thread_id

    def _build_args_schema(self, args_schema: dict) -> type[BaseModel]:
        OriginalModel = create_model(args_schema)
        class ScreenshotPersistingToolArgs(OriginalModel):
            tool_call_id: Annotated[str, InjectedToolCallId]
        return ScreenshotPersistingToolArgs

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def _arun(self, *args: Any, config: RunnableConfig,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: Any) -> Any:
        result = await self._mcp_tool._arun(*args, config=config, run_manager=run_manager, **kwargs)
        try:
            path = self._extract_screenshot_path(result)
            file = await self._save_screenshot_to_database(path)
            return ToolMessage(
                tool_call_id=kwargs.get("tool_call_id"),
                name=self.name,
                content=self._replace_file_path_references(result[0], path, file),
                response_metadata={"file": file}
            )
        except Exception:
            logger.error("Problem saving screenshot to database", exc_info=True)
            return "Error saving screenshot to database"

    def _extract_screenshot_path(self, result: Any) -> str:
        match = re.search(r"(?:/tmp/playwright-output|\.playwright-mcp)/[^\s)]+\.png", result[0])
        if not match:
            raise ValueError("Screenshot path not found in tool output")
        path = match.group(0)
        if path.startswith(".playwright-mcp/"):
            return f"{self._PLAYWRIGHT_OUTPUT_DIR}/{path}"
        return path

    async def _save_screenshot_to_database(self, file_path: Any) -> FileMetadata:
        host_path = file_path.replace(self._PLAYWRIGHT_OUTPUT_DIR, env.browser_tool_playwright_output_dir)
        file_name = host_path.split("/")[-1]
        async with aiofiles.open(host_path, "rb") as f:
            ret = await FileRepository(self._db).add(File(name=file_name,
                content_type="image/png",
                user_id=self._user_id,
                content=await f.read(),
                status=FileStatus.PROCESSED))
            await aiofiles.os.remove(host_path)
            return FileMetadata.from_file(ret)

    def _replace_file_path_references(self, content: str, path: str, file: FileMetadata) -> str:
        final_path = f"/chat/{self._thread_id}/files/{file.id}"
        ret = content.replace(path, f"{env.frontend_url}{final_path}", 1)
        return ret.replace(path, final_path, 1)


class BrowserTool(AgentTool):
    id: str = BROWSER_TOOL_ID
    name: str = "Browser Tools"
    description: str = (
        "Provides tools to control a browser like navigating, click elements, fill, get page contents, screenshots, etc."
    )
    config_schema: dict = load_schema(__file__)
    _tools: Optional[list[BaseTool]] = None

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        pass

    async def teardown(self):
        pass

    @asynccontextmanager
    async def load(self) -> AsyncIterator['BrowserTool']:
        server_name = "playwright"
        client = MultiServerMCPClient({server_name: {"transport": "streamable_http", "url": env.browser_tool_playwright_mcp_url }})
        async with client.session(server_name) as mcp_session:
            self._tools = await load_mcp_tools(mcp_session)
            self._tools = [ScreenshotPersistingTool(cast(StructuredTool, t), self.user_id, self._thread_id, cast(AsyncSession, self._db)) if t.name == "browser_take_screenshot" else t for t in self._tools]
            for tool in self._tools:
                tool.callbacks = [StatusUpdateCallbackHandler(tool.name, description=tool.description)]
            yield self

    async def clone(
        self,
        agent_id: int,
        cloned_agent_id: int,
        tool_id: str,
        user_id: int,
        db: AsyncSession,
    ) -> None:
        pass

    async def build_langchain_tools(self) -> List[BaseTool]:
        if not self._tools:
            raise RuntimeError("Browser tool has not been set up properly")
        return self._tools
