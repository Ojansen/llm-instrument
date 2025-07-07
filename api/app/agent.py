import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

llm_base_url = os.getenv('OLLAMA_URL')
if not llm_base_url:
    print('OLLAMA_URL not set using fallback http://host.docker.internal:1234/v1')
    llm_base_url = 'http://host.docker.internal:1234/v1'


model = os.getenv('OLLAMA_MODEL')
if not model:
    print('OLLAMA_MODEL no model defined')
    exit(1)


ollama_model = OpenAIModel(
    model_name=model,
    provider=OpenAIProvider(base_url=llm_base_url)
)
agent = Agent(
    ollama_model,
    system_prompt=""
)