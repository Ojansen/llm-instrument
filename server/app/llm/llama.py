from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.lmstudio import LMStudio
import os


class Llama:
    def __init__(self, system_prompt="", temperature=0.8) -> None:
        self.system_prompt = system_prompt
        self.temperature = temperature

        self.model_name = os.getenv("OLLAMA_MODEL") or "llama-3.2-3b-instruct"
        self.embedding_model = (
            os.getenv("OLLAMA_EMBEDDING_MODEL") or "text-embedding-bge-m3"
        )
        self.llm_base_url = (
            os.getenv("OLLAMA_URL") or "http://host.docker.internal:1234/v1"
        )
        self.agent = LMStudio(
            model_name=self.model_name,
            base_url=self.llm_base_url,
            temperature=self.temperature,
        )

    def inference(self, prompt: str) -> str:
        response = self.agent.complete(prompt)
        return str(response)

    def embed(self, text: str):
        embed_model = OpenAILikeEmbedding(
            model_name=self.embedding_model,
            api_base=self.llm_base_url,
        )
        return embed_model.get_text_embedding(text)
