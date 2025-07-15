from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider
from openai import AsyncAzureOpenAI
from supabase import Client
from typing import List
from utils import get_azure_model

load_dotenv(override=True)
model = get_azure_model()

source = os.getenv('SOURCE', 'ROG-Strix-G16')
print(f"Using source: {source}")

logfire.configure(send_to_logfire=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncAzureOpenAI

system_prompt = """
Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncAzureOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # 取得用户查询的Embedding
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # 查詢Supabase以獲取最相關的文檔
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 1, # how many rows to return
                'filter': {'source': source}
            }
        ).execute()
        print("retrieve_relevant_documentation")
        # print(f"Query result: {result.data}")
        if not result.data:
            return "No relevant documentation found."
            
        # 整理文檔內容
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
            # {doc['title']}

            {doc['content']}
            """
            formatted_chunks.append(chunk_text)
            
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"
    
@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        print("list_documentation_pages")
        # 查詢Supabase以獲取所有該文檔頁面
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title') \
            .eq('metadata->>source', source) \
            .execute()
        
        if not result.data:
            return []

        titles = sorted(set(doc['title'] for doc in result.data))
        print(f"Found {len(titles)} documentation pages.")
        return titles
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], title: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        print(f"get_page_content for title: {title}")

        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('title', title) \
            .eq('metadata->>source', source) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for title: {title}"
            
        page_title = result.data[0]['title'].split(' - ')[0] 
        formatted_content = [f"# {page_title}\n"]

        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
