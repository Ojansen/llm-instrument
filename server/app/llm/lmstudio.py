import os

from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class LmStudio:
    def __init__(self, system_prompt="") -> None:
        self.system_prompt = system_prompt
        self.model_name = os.getenv("OLLAMA_MODEL") or "llama-3.2-3b-instruct"
        self.embedding_model = (
            os.getenv("OLLAMA_EMBEDDING_MODEL") or "text-embedding-bge-m3"
        )
        self.llm_base_url = (
            os.getenv("OLLAMA_URL") or "http://host.docker.internal:1234/v1"
        )
        self.agent = Agent(
            OpenAIModel(
                model_name=self.model_name,
                provider=OpenAIProvider(base_url=self.llm_base_url),
            ),
            system_prompt=self.system_prompt,
        )

    def inference(self, prompt: str) -> str:
        return self.agent.run_sync(prompt).output

    def embed(self, text: str):
        embed_model = OpenAILikeEmbedding(
            model_name=self.embedding_model,
            api_base=self.llm_base_url,
        )
        return embed_model.get_text_embedding(text)
