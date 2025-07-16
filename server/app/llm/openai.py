import os

from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai import OpenAI


class OpenAILLM:
    def __init__(self, system_prompt="You are helpful", temperature=0.8):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_name = "gpt-4o-mini"
        self.llm_base_url = "https://api.openai.com/v1"
        self.embedding_dimension = 1024
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = OpenAILikeEmbedding(
            model_name="text-embedding-3-small",
            api_key=self.api_key,
            api_base=self.llm_base_url,
            dimensions=self.embedding_dimension,
        )
        self.agent = OpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.llm_base_url,
        )

    def __repr__(self):
        return f"OpenAI(model_name={self.model_name}, temperature={self.temperature})"

    async def inference(self, prompt: str):
        response = await self.agent.acomplete(prompt)
        return response.text

    async def embed(self, text: str):
        return await self.embedding_model.aget_text_embedding(text)
