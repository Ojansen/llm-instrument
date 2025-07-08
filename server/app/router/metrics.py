from typing import Dict, Any, Protocol, Union
import logfire

from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    CorrectnessEvaluator,
)
from llama_index.core.llms import CustomLLM
from pydantic_ai import Agent


class AgentInterface(Protocol):
    agent: Union[Agent, CustomLLM]

    def inference(self, prompt: str) -> str:
        pass


class Metrics:
    def __init__(self, llm: AgentInterface):
        self._llm = llm

    def cosine_similarity(self, prompt: str, reference: str):
        evaluator = SemanticSimilarityEvaluator()
        result_sync = self._llm.inference(prompt=prompt)
        result = evaluator.evaluate(response=result_sync, reference=reference)

        with logfire.span("Cosine Similarity"):
            logfire.info(result.feedback)

        return {"score": result.score, "passing": result.passing, "output": result_sync}

    def correctness(self, prompt: str, reference: str):
        evaluator = CorrectnessEvaluator()
        prompt_output = self._llm.inference(prompt=prompt)
        result = evaluator.evaluate(
            query=prompt,
            response=prompt_output,
            reference=reference,
        )
        print(result.feedback)
        return {"score": result.score, "passing": result.passing}
