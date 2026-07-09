import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import json
import logging
from typing import Callable, List, Any, cast, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.messages.utils import _is_message_type
from langchain_core.tools import tool, BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.domain import Agent
from ..agents.repos import AgentToolConfigRepository
from ..ai_models import ai_factory
from ..ai_models.repos import AiModelRepository
from ..core.env import env
from ..threads.core import trim_messages_to_fit_model
from ..tools.core import AgentTool, AgentToolMetadata
from ..tools.repos import ToolRepository
from ..usage.domain import MessageUsage
from .domain import ThreadMessage, ThreadMessageOrigin, MAX_THREAD_NAME_LENGTH, AgentEvent, AgentActionEvent, AgentFileEvent, AgentMessageEvent, AgentAction


logger = logging.getLogger(__name__)


# adding this tool because we are going to add more tools in the future and right now
# is easier to add a lame tool and make it work with it than without any tools
@tool
def clock() -> str:
    """Returns the current time in UTC."""
    return f"{datetime.now(timezone.utc)}."


class AgentEngine:
    _MEMORY_INPUT_KEY = "input"

    def __init__(self, agent: Agent, user_id: int, db: AsyncSession):
        self._agent = agent
        self._user_id = user_id
        self._db = db

    async def load_tools(self, stack: AsyncExitStack, thread_id: Optional[int] = None) -> List[AgentTool]:
        tool_configs = await AgentToolConfigRepository(self._db).find_by_agent_id(
            agent_id=self._agent.id
        )
        ret = []
        for tc in tool_configs:
            agent_tool = ToolRepository().find_by_id(tc.tool_id)
            if not agent_tool:
                raise ValueError(f"Tool {tc.tool_id} not found")
            agent_tool.configure(self._agent, self._user_id, tc.config, self._db, thread_id=thread_id)
            tool = await stack.enter_async_context(agent_tool.load())
            ret.append(tool)
        logger.debug(f"[engine] Tools loaded for agent_id={self._agent.id}: {[t.id for t in ret]}")
        return ret

    async def answer(self, messages: List[ThreadMessage], message_usage: MessageUsage, stop_event: asyncio.Event) -> AsyncIterator[AgentEvent]:
        logger.info(f"[engine] Answering thread_id={messages[0].thread_id} agent_id={self._agent.id} messages={len(messages)}")
        llm = ai_factory.build_streaming_chat_model(self._agent.model.id, self._agent.model_temperature,  self._agent.model_reasoning_effort)
        async with AsyncExitStack() as stack:
            agent_tools = await self.load_tools(stack, thread_id=messages[0].thread_id)
            tools = [ lt for t in agent_tools for lt in await t.build_langchain_tools() ]
            tools.append(clock)
            # Enable error handling so ToolException from MCP tools (execution errors)
            # are shown to the LLM instead of crashing the agent
            agent = create_react_agent(
                llm, ToolNode(tools, handle_tool_errors=True), pre_model_hook=self._build_message_trimmer(llm, tools)
            )

            input = self._build_input(messages)
            generated_content = ""
            stream = agent.astream(
                input,
                {
                    "recursion_limit": self._agent.recursion_limit
                },
                stream_mode=["updates", "messages", "custom"],
            )
            async for mode, content in stream:
                if stop_event.is_set():
                    break
                if mode == "updates":
                    async for status_update in self._process_updates(content):
                        yield status_update
                elif mode == "custom":
                    yield cast(AgentActionEvent, content)
                elif mode == "messages":
                    msg, metadata = content
                    metadata = cast(dict, metadata)
                    # we need to filter AI messages since AI messages from tools are also returned
                    if ((isinstance(msg, AIMessage) and metadata.get("langgraph_node") != "tools") \
                        or (isinstance(msg, ToolMessage) and msg.response_metadata.get("return_direct"))) \
                        and msg.content:
                        content = self._get_content(msg.content)
                        generated_content += content
                        yield AgentMessageEvent(content=content)
                    if isinstance(msg, AIMessage):
                        message_usage.increment_with_metadata(msg.usage_metadata, self._agent.model)
                    elif isinstance(msg, ToolMessage):
                        agent_tool_metadata = AgentToolMetadata.model_validate(msg.response_metadata)
                        message_usage.increment_tool_usage(agent_tool_metadata.tool_usage)
                        if agent_tool_metadata.file:
                            yield AgentFileEvent(file=agent_tool_metadata.file)

            logger.info(f"[engine] Stream finished thread_id={messages[0].thread_id} input_tokens={message_usage.prompt_usage.quantity} output_tokens={message_usage.completion_usage.quantity}")
            # If the response was stopped, approximate the token usage
            if stop_event.is_set():
                approximate_input_tokens = llm.get_num_tokens_from_messages(input["messages"]) + self._count_tools_tokens(tools, llm)
                approximate_output_tokens = llm.get_num_tokens(generated_content) if generated_content else 0
                message_usage.increment_with_metadata(
                    {
                        "input_tokens": approximate_input_tokens,
                        "output_tokens": approximate_output_tokens,
                        "total_tokens": approximate_input_tokens + approximate_output_tokens
                    }, self._agent.model)

    def _get_content(self, msg: str | list[str | dict]) -> str:
        if isinstance(msg, str):
            return msg
        if isinstance(msg, list) and msg:
            texts = []
            for item in msg:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        texts.append(text)
            return "".join(texts)
        raise ValueError(f"Invalid message type: {type(msg)}")

    async def _process_updates(self, content: Any) -> AsyncIterator[AgentActionEvent]:
        if isinstance(content, dict):
            ((key, value), *_) = content.items()
            if key == "pre_model_hook":
                yield AgentActionEvent(action=AgentAction.PRE_MODEL_HOOK)
            elif key == "agent":
                async for update in self._process_agent(value):
                    yield update
        else:
            try:
                json_content = json.dumps(content, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                json_content = str(content)
            yield AgentActionEvent(action=AgentAction.UNDEFINED, result=json_content)

    async def _process_agent(self, value: Any) -> AsyncIterator[AgentActionEvent]:
        if not value.get("messages"):
            return
        message = value["messages"][0]
        finish_reason = message.response_metadata.get("finish_reason")

        if finish_reason != "stop":
            if hasattr(message, "tool_calls") and message.tool_calls:
                result = []
                for tool_calls in message.tool_calls:
                    result.append(tool_calls["name"])
                yield AgentActionEvent(action=AgentAction.PLANNING, result=result)

    def _build_message_trimmer(
        self, llm: BaseChatModel, tools: List[BaseTool]
    ) -> Callable[[Any], Any]:
        def pre_model_hook(state):
            # this is mostly the same logic (but simplified) as invoking langchain trim_messages with last strategy and allow partial
            # , but if a message is too big to fit, instead of returning the last part, we return the first part of the message.
            # This way, we keep the first part of the message that we consider should be more relevan.
            # For example, if user sends text and files, then text is kept, first files are kept, and the first part of the last file that fits is kept as well.
            messages = state["messages"]
            system_message = messages[0]
            messages = messages[1:]
            token_counter = llm.get_num_tokens_from_messages

            # Reverse messages to use _first_max_tokens with reversed logic
            messages = messages[::-1]

            end_index = next(
                i
                for i, x in enumerate(messages)
                if _is_message_type(x, (HumanMessage, ToolMessage))
            )
            messages = messages[end_index:]

            tools_tokens = self._count_tools_tokens(tools, llm)
            system_message_tokens = token_counter([system_message])
            reserved_tokens = tools_tokens + system_message_tokens

            result = trim_messages_to_fit_model(
                messages,
                token_counter=token_counter,
                model=self._agent.model,
                reserved_tokens=reserved_tokens,
                end_on=HumanMessage,
            )
            # Re-reverse the messages and add back the system message
            return {"llm_input_messages": [system_message] + result[::-1]}

        return pre_model_hook

    def _count_tools_tokens(self, tools: List[BaseTool], llm: BaseChatModel) -> int:
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        tools_json = json.dumps(openai_tools)
        return llm.get_num_tokens(tools_json)

    def _build_input(self, messages: List[ThreadMessage]) -> Any:
        messages_list: List[BaseMessage] = [SystemMessage(self._agent.system_prompt)]
        for message in messages:
            if message.origin == ThreadMessageOrigin.USER:
                content = []
                message_text = message.text

                for file_obj in message.files:
                    # svg files should be treated as text
                    if file_obj.file.content_type.startswith("image/") and not file_obj.file.name.lower().endswith('.svg'):
                        content.append(
                            {
                                "type": "image",
                                "source_type": "base64",
                                "mime_type": file_obj.file.content_type,
                                "data": base64.b64encode(file_obj.file.content).decode(
                                    "utf-8"
                                ),
                            }
                        )
                    else:
                        message_text = (
                            message_text
                            + "\n\n File named: "
                            + file_obj.file.name
                            + "\n\n"
                            + file_obj.file.processed_content
                            if file_obj.file.processed_content
                            else ""
                        )

                if message_text.strip():
                    content.append({"type": "text", "text": message_text})

                messages_list.append(HumanMessage(content=content))
            else:
                messages_list.append(AIMessage(message.text))
        return {"messages": messages_list}


async def build_thread_name(first_thread_message: str, message_usage: MessageUsage, db: AsyncSession) -> str:
    model = await AiModelRepository(db).find_by_id(env.internal_generator_model)
    if not model:
        raise ValueError("Internal generator model not found")
    llm = ai_factory.build_chat_model(model.id, env.internal_generator_temperature)
    system_prompt = "From the following user message generate a short (less than 80 characters) title for the chat. Do not include quoting or any special characters."
    # invoke the llm using the prompt as system prompt and the first thread message as user message
    response = await llm.ainvoke(
        [SystemMessage(system_prompt), HumanMessage(first_thread_message)]
    )
    response = cast(AIMessage, response)
    message_usage.increment_with_metadata(response.usage_metadata, model)
    return cast(str, response.content)[:MAX_THREAD_NAME_LENGTH].replace("\n", " ")
