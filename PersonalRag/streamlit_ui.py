from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncAzureOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

AZURE_KEY = os.getenv("AZURE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_LLM = os.getenv("AZURE_LLM")
AZURE_VERSION = os.getenv("AZURE_VERSION")

openai_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_VERSION,
    api_key=AZURE_KEY
)

supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # pydantic ai dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    print(len(st.session_state.messages))

    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # 暫時放全部內容
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        # 擷取流式回應
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # 排除掉 user-prompt 的部分
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # 加入最終回應
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.title("Agentic RAG")
    st.write("Ask any question.")

    # 清除歷史紀錄
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 根據 ModelRequest 和 ModelResponse 類型顯示訊息
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("What questions do you have?")

    if user_input:
        # 新增回應到訊息列表
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # 顯示使用者輸入的訊息
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
