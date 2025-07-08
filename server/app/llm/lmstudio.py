import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class LmStudio:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.model_name = os.getenv("OLLAMA_MODEL") or "llama-3.2-3b-instruct"
        self.llm_base_url = (
            os.getenv("OLLAMA_URL") or "http://host.docker.internal:1234/v1"
        )

    def agent(self):
        return Agent(
            OpenAIModel(
                model_name=self.model_name,
                provider=OpenAIProvider(base_url=self.llm_base_url),
            ),
            system_prompt=self.system_prompt,
        )
