import logging
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from sqlmodel import select
from testcontainers.generic import ServerContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from .common import *

from tero.agents.api import AGENT_TOOL_FILE_PATH
from tero.tools.browser import BrowserTool, BROWSER_TOOL_ID
from tero.tools.docs import DocsTool, DOCS_TOOL_ID
from tero.tools.docs import tool as docs_tool_module
from tero.tools.docs.tool import DocsExecutionStep, DocsStatusUpdateCallbackHandler, DocumentUrlSolvingRetriever
from tero.tools.docs.repos import DocToolFileRepository
from tero.tools.jira import JiraTool
from tero.tools.redmine import RedmineTool
from tero.tools.github import GitHubTool
from tero.tools.youtrack import YouTrackTool
from tero.tools.practitest import PractiTestTool
from tero.tools.mcp import McpTool
from tero.tools.web import WebTool, WEB_TOOL_ID
from tero.usage.domain import Usage, UsageType


logger = logging.getLogger(__name__)


@pytest.fixture
def stub_web_tool_tavily_ainvoke():
    search = '{"results": [{"url": "https://example.com", "content": "Stub search result."}]}'
    extract = [{"url": "https://example.com", "raw_content": "Stub extracted body."}]
    with (
        patch("tero.tools.web.tool.TavilySearch.ainvoke", new=AsyncMock(return_value=search)),
        patch("tero.tools.web.tool.TavilyExtract.ainvoke", new=AsyncMock(return_value=extract)),
    ):
        yield


async def test_find_tools(client: AsyncClient, session: AsyncSession):
    resp = await client.get(f"{BASE_PATH}/tools")
    expected_tools = [DocsTool(), McpTool(), JiraTool(), BrowserTool(), GitHubTool(), YouTrackTool(), PractiTestTool(), RedmineTool()]
    if env.web_tool_tavily_api_key or (env.web_tool_google_api_key and env.web_tool_google_custom_search_engine_id):
        expected_tools.append(WebTool())
    expected_tools.sort(key=lambda t: t.name.casefold())
    assert_response(resp, expected_tools)


async def test_docs_tool(client: AsyncClient):
    await _configure_docs_tool_with_file("Emma's routine.pdf", client)
    answer = await _answer_question("What time does Emma wake up according to the document? Output only the time in H:MM format. Don't use clock tool.", client)
    assert "7:35" in answer


async def _configure_docs_tool_with_file(file_path: str, client: AsyncClient) -> int:
    await configure_agent_tool(AGENT_ID, DOCS_TOOL_ID, {"advancedFileProcessing": False}, client)
    content = await find_asset_bytes(file_path)
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client, filename=os.path.basename(file_path),
                                               content=content)
    await await_files_processed(AGENT_ID, DOCS_TOOL_ID, file_id, client)
    return file_id


async def _answer_question(question: str, client: AsyncClient) -> str:
    resp = await create_thread(AGENT_ID, client)
    resp.raise_for_status()
    thread_id = resp.json()["id"]
    async with add_message_to_thread(client, thread_id, question) as resp:
        resp.raise_for_status()
        response = ""
        async for event in resp.aiter_text():
            response += _parse_event(event)
        return response


def _parse_event(event_str: str) -> str:
    event = None
    data = ""
    event_marker = "event: "
    data_marker = "data: "
    for line in event_str.splitlines():
        if line.startswith(data_marker):
            part = line[len(data_marker):]
            data += (part if part else "\n")
        elif line.startswith(event_marker):
            event = line[len(event_marker):]
    if event == "error":
        raise Exception(data)
    return data


async def test_docs_tool_with_removed_file(client: AsyncClient):
    file_id = await _configure_docs_tool_with_file("Emma's routine.pdf", client)
    resp = await client.delete(AGENT_TOOL_FILE_PATH.format(agent_id=1, tool_id=DOCS_TOOL_ID, file_id=file_id))
    resp.raise_for_status()
    answer = await _answer_question("What time does Emma wake up according to the document? Output only the time in H:MM format. Don't use clock tool.", client)
    assert "7:35" not in answer


@pytest.mark.usefixtures("stub_web_tool_tavily_ainvoke")
async def test_web_tool_search_usage(client: AsyncClient, session: AsyncSession):
    await configure_agent_tool(AGENT_ID, WEB_TOOL_ID, {}, client)

    initial_usage = await session.exec(select(Usage).where(Usage.type == UsageType.WEB_SEARCH))
    initial_count = len(initial_usage.all())

    await _answer_question("Search for the latest news about AI", client)

    final_usage = await session.exec(select(Usage).where(Usage.type == UsageType.WEB_SEARCH))
    final_count = len(final_usage.all())

    assert final_count > initial_count


@pytest.mark.usefixtures("stub_web_tool_tavily_ainvoke")
async def test_web_tool_extract_usage(client: AsyncClient, session: AsyncSession):
    await configure_agent_tool(AGENT_ID, WEB_TOOL_ID, {}, client)

    initial_usage = await session.exec(select(Usage).where(Usage.type == UsageType.WEB_EXTRACT))
    initial_count = len(initial_usage.all())

    await _answer_question("What is the first paragraph of this url: https://modelcontextprotocol.io/", client)

    final_usage = await session.exec(select(Usage).where(Usage.type == UsageType.WEB_EXTRACT))
    final_count = len(final_usage.all())

    assert final_count > initial_count


@pytest.fixture(scope="function")
def containers_network() -> Generator[Network, None, None]:
    with Network() as network:
        yield network


@pytest.fixture(scope="function")
def playwright_container_url(containers_network: Network) -> Generator[str, None, None]:
    port = 8931
    output_dir = "/tmp/share/playwright-output"
    env.browser_tool_playwright_output_dir = output_dir
    with DockerContainer("mcp/playwright")\
            .with_exposed_ports(port)\
            .with_network(containers_network)\
            .with_kwargs(user="0:0")\
            .with_command([f"--port={port}", "--allowed-hosts=*", "--host=0.0.0.0", "--output-dir=/tmp/playwright-output"]) \
            .with_volume_mapping(output_dir, "/tmp/playwright-output", "rw") \
            .waiting_for(LogMessageWaitStrategy(f"Listening on http://localhost:{port}")) \
            as container:
        yield f"http://{container.get_container_host_ip()}:{container.get_exposed_port(port)}"


@pytest.fixture(scope="function")
def nginx_container_url(containers_network: Network) -> Generator[str, None, None]:
    nginx_port = 80
    network_alias = "nginx"
    with ServerContainer(nginx_port, "nginx:1.29") \
            .with_network(containers_network) \
            .with_network_aliases(network_alias):
        yield f"http://{network_alias}:{nginx_port}"


async def test_browser_tool(client: AsyncClient, playwright_container_url: str, nginx_container_url: str):
    await _configure_browser_tool(playwright_container_url, client)
    answer = await _answer_question(f"Navigate to {nginx_container_url} and get body of the page", client)
    assert "Welcome to nginx!" in answer


async def _configure_browser_tool(playwright_container_url: str, client: AsyncClient):
    env.browser_tool_playwright_mcp_url = playwright_container_url + "/mcp"
    await configure_agent_tool(AGENT_ID, BROWSER_TOOL_ID, {}, client)


async def test_browser_tool_screenshot(client: AsyncClient, playwright_container_url: str, nginx_container_url: str):
    await _configure_browser_tool(playwright_container_url, client)
    answer = await _answer_question(f"Navigate to {nginx_container_url} and take a screenshot of the page", client)
    assert '"files": [{' in answer


async def test_docs_tool_skip_descriptions_true(client: AsyncClient, session: AsyncSession):
    """REQ-INDEX-1: skipDescriptions=true skips description generation, indexing still runs."""
    await configure_agent_tool(AGENT_ID, DOCS_TOOL_ID,
                               {"skipDescriptions": True, "advancedFileProcessing": False}, client)
    content = await find_asset_bytes("Emma's routine.pdf")
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client,
                                                   filename="Emma's routine.pdf", content=content)
    await await_files_processed(AGENT_ID, DOCS_TOOL_ID, file_id, client)

    # Verify file is retrievable (aindex ran)
    answer = await _answer_question(
        "What time does Emma wake up according to the document? Output only the time in H:MM format. Don't use clock tool.",
        client)
    assert "7:35" in answer, "File should be indexed and retrievable even with skipDescriptions=true"

    # Verify no DocToolFile records were created (description generation skipped)
    doc_files = await DocToolFileRepository(session).find_by_agent_id(AGENT_ID)
    assert len(doc_files) == 0, f"Expected 0 DocToolFile records with skipDescriptions=true, got {len(doc_files)}"


async def test_docs_tool_skip_descriptions_false_generates_descriptions(client: AsyncClient, session: AsyncSession):
    """REQ-INDEX-1: skipDescriptions=false generates descriptions (same as absent key)."""
    await configure_agent_tool(AGENT_ID, DOCS_TOOL_ID,
                               {"skipDescriptions": False, "advancedFileProcessing": False}, client)
    content = await find_asset_bytes("Emma's routine.pdf")
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client,
                                                   filename="Emma's routine.pdf", content=content)
    await await_files_processed(AGENT_ID, DOCS_TOOL_ID, file_id)

    # Verify DocToolFile records exist (description generation ran)
    doc_files = await DocToolFileRepository(session).find_by_agent_id(AGENT_ID)
    assert len(doc_files) > 0, "Expected DocToolFile records when skipDescriptions=false"


async def test_docs_tool_skip_descriptions_key_absent(client: AsyncClient, session: AsyncSession):
    """REQ-INDEX-1: Config key absent defaults to false — descriptions run."""
    await configure_agent_tool(AGENT_ID, DOCS_TOOL_ID,
                               {"advancedFileProcessing": False}, client)
    content = await find_asset_bytes("Emma's routine.pdf")
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client,
                                                   filename="Emma's routine.pdf", content=content)
    await await_files_processed(AGENT_ID, DOCS_TOOL_ID, file_id)

    # Verify DocToolFile records exist (description generation ran — default is false)
    doc_files = await DocToolFileRepository(session).find_by_agent_id(AGENT_ID)
    assert len(doc_files) > 0, "Expected DocToolFile records when key absent (defaults to false)"


# Former server-side preview length; retrieved payloads must now be lossless and
# presentation truncation moved to the UI layer (textPreview.ts).
DOCS_TOOL_PREVIEW_CUTOFF = 150
DOCS_TOOL_LONG_CHUNK = (
    "Emma wakes up at 7:35 and follows a strict morning routine: she prepares "
    "breakfast, reviews the day schedule, walks the dog, and plans her tasks before "
    "starting work. This chunk is intentionally longer than the former 150-character "
    "server-side preview cutoff so truncation cannot hide retrieved content."
)
DOCS_TOOL_SHORT_CHUNK = "Short chunk without truncation."


def _docs_tool_documents() -> list[Document]:
    return [
        Document(page_content=DOCS_TOOL_LONG_CHUNK, metadata={"id": "1"}),
        Document(page_content=DOCS_TOOL_SHORT_CHUNK, metadata={"id": "2"}),
    ]


def test_docs_tool_retrieved_payload_full_content():
    """R1 full content emitted: payload keeps complete content with no server-side ellipsis."""
    documents = _docs_tool_documents()
    assert len(DOCS_TOOL_LONG_CHUNK) > DOCS_TOOL_PREVIEW_CUTOFF

    payload = docs_tool_module._build_retrieved_payload(documents)

    assert payload == [DOCS_TOOL_LONG_CHUNK, DOCS_TOOL_SHORT_CHUNK]


def test_docs_tool_retrieved_payload_stays_list_of_strings():
    """R1 payload shape unchanged: still list[str] for SSE and eval consumers."""
    boundary_chunk = "x" * DOCS_TOOL_PREVIEW_CUTOFF
    multiline_chunk = "Line one\nLine two, section: Retrieval\nCaf\u00e9 r\u00e9sum\u00e9 \u2713"
    documents = [
        Document(page_content=boundary_chunk, metadata={"id": "10"}),
        Document(page_content=multiline_chunk, metadata={"id": "11"}),
    ]

    payload = docs_tool_module._build_retrieved_payload(documents)

    assert payload == [boundary_chunk, multiline_chunk]
    assert isinstance(payload, list)
    assert all(isinstance(chunk, str) for chunk in payload)


async def test_docs_tool_retrieved_payload_event_emits_full_content():
    """R1 full content emitted: on_retriever_end streams the complete chunks as a list."""
    documents = _docs_tool_documents()
    writer = MagicMock()

    with patch("tero.tools.docs.tool.get_stream_writer", return_value=writer):
        await DocsStatusUpdateCallbackHandler(DOCS_TOOL_ID, "Docs").on_retriever_end(documents)

    event = writer.call_args.args[0]
    assert event.step == DocsExecutionStep.RETRIEVED
    assert event.result == [DOCS_TOOL_LONG_CHUNK, DOCS_TOOL_SHORT_CHUNK]


def test_docs_tool_retriever_default_k5_without_rerank():
    """R3 disabled by default: plain retriever queried with k=5, no rerank stage (approval)."""
    tool = DocsTool()
    tool.configure(MagicMock(id=7), user_id=1, config={}, db=MagicMock())

    with patch.object(DocsTool, "_build_vectorstore", return_value=MagicMock(spec=VectorStore)):
        retriever = tool._build_retriever()

    assert type(retriever) is DocumentUrlSolvingRetriever
    assert retriever.search_kwargs == {"k": 5}
