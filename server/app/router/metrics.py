from typing import Dict, Any
import logfire

from llama_index.core.evaluation import SemanticSimilarityEvaluator
from pydantic_ai import Agent

from app.router.inference import Inference


class Metrics:
    def __init__(self, agent: Agent):
        self.agent = agent

    def cosine_similarity(self, prompt: str, reference: str):
        evaluator = SemanticSimilarityEvaluator()
        result_sync = Inference(agent=self.agent).run(prompt=prompt)
        result = evaluator.evaluate(response=result_sync, reference=reference)

        with logfire.span("Cosine Similarity"):
            logfire.span(result.feedback)

        return {"score": result.score, "passing": result.passing, "output": result_sync}
