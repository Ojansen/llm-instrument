from enum import Enum
from typing import Protocol, Union

from llama_index.core.llms import CustomLLM
from pydantic_ai import Agent


class AgentInterface(Protocol):
    agent: Union[Agent, CustomLLM]

    def inference(self, prompt: str) -> str:
        pass

    def embed(self, text: str) -> list[float]:
        pass


class MetricType(str, Enum):
    similarity = "similarity"
    correctness = "correctness"
    faithfulness = "faithfulness"
