from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.lmstudio import LMStudio
import os


class Llama:
    def __init__(self, system_prompt="You are helpful", temperature=0.8) -> None:
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_name = os.getenv("OLLAMA_MODEL") or "llama-3.2-3b-instruct"
        self.llm_base_url = (
            os.getenv("OLLAMA_URL") or "http://host.docker.internal:1234/v1"
        )
        self.api_key = "lm-studio"
        self.embedding_dimension = 1024
        self.embedding_model = OpenAILikeEmbedding(
            model_name=os.getenv("OLLAMA_EMBEDDING_MODEL") or "text-embedding-bge-m3",
            api_base=self.llm_base_url,
            dimension=self.embedding_dimension,
        )
        self.agent = LMStudio(
            model_name=self.model_name,
            base_url=self.llm_base_url,
            temperature=self.temperature,
            request_timeout=120.0,
        )

    def __repr__(self):
        return f"Llama(model_name={self.model_name}, temperature={self.temperature})"

    async def inference(self, prompt: str) -> str:
        response = await self.agent.acomplete(prompt)
        return response.text

    async def embed(self, text: str):
        return await self.embedding_model.aget_text_embedding(text)
