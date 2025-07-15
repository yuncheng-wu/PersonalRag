from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_azure_model():
    llm = os.getenv('AZURE_LLM')
    base_url = os.getenv('AZURE_ENDPOINT')
    api_key = os.getenv('AZURE_KEY')
    api_version = os.getenv('AZURE_VERSION')

    return OpenAIModel(
        llm,
        provider=AzureProvider(
            azure_endpoint=base_url,
            api_version=api_version,
            api_key=api_key,
        ),
    )

def get_ollama_model():
    llm = os.getenv('OLLAMA_MODEL')
    base_url = os.getenv('OLLAMA_ENDPOINT')

    return OpenAIModel(
        model_name= llm,
        provider=OpenAIProvider(
            base_url=base_url
        )
    )