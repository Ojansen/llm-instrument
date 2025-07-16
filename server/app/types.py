from enum import Enum
from typing import Protocol, Union, Coroutine, Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import CustomLLM


class AgentInterface(Protocol):
    system_prompt: str
    temperature: float
    model_name: str
    llm_base_url: str
    api_key: str
    embedding_dimension: int
    embedding_model = BaseEmbedding | None
    agent: CustomLLM

    async def inference(self, prompt: str):
        pass

    async def embed(self, text: str):
        pass


class MetricType(str, Enum):
    similarity = "similarity"
    correctness = "correctness"
    faithfulness = "faithfulness"
    ragchecker = "ragchecker"
