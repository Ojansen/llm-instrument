from enum import Enum
from typing import Protocol, Union

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import CustomLLM
from pydantic_ai import Agent


class AgentInterface(Protocol):
    agent: Union[Agent, CustomLLM]
    model_name: str
    llm_base_url: str
    embedding_model = BaseEmbedding | None

    def inference(self, prompt: str) -> str:
        pass

    def embed(self, text: str) -> list[float]:
        pass


class MetricType(str, Enum):
    similarity = "similarity"
    correctness = "correctness"
    faithfulness = "faithfulness"
