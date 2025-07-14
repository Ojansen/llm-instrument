from typing import Dict, Any, Protocol, Union
import logfire

from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
)
from llama_index.core.schema import BaseNode
from ragchecker import RAGChecker, RAGResults
from ragchecker.integrations.llama_index import response_to_rag_results
from ragchecker.metrics import all_metrics

from app.types import AgentInterface
from app.utils.vector_store import VectorStore


class Metrics:
    def __init__(self, llm: AgentInterface):
        self._llm = llm
        self._vector_store = VectorStore(llm)

    def __repr__(self):
        return f"Metrics({self._llm}, {self._vector_store})"

    @logfire.instrument
    def cosine_similarity(self, prompt: str, reference: str):
        evaluator = SemanticSimilarityEvaluator(embed_model=self._llm.embedding_model)
        prompt_output = self._llm.inference(prompt=prompt)
        result = evaluator.evaluate(
            query=prompt, response=prompt_output, reference=reference
        )

        with logfire.span("Cosine Similarity"):
            logfire.info(
                "result",
                attributes={
                    "prompt": prompt,
                    "response": prompt_output,
                    "reference": reference,
                    "feedback": result.feedback,
                },
            )

        return result

    @logfire.instrument
    def correctness(self, prompt: str, reference: str):
        evaluator = CorrectnessEvaluator(llm=self._llm.agent)
        prompt_output = self._llm.inference(prompt=prompt)
        result = evaluator.evaluate(
            query=prompt,
            response=prompt_output,
            reference=reference,
        )
        with logfire.span("Correctness"):
            logfire.info(
                "result",
                attributes={
                    "prompt": prompt,
                    "response": prompt_output,
                    "reference": reference,
                    "feedback": result.feedback,
                },
            )
        return result

    @logfire.instrument
    def faithfulness(self, prompt: str):
        evaluator = FaithfulnessEvaluator(llm=self._llm.agent)
        response_vector = self._vector_store.query(prompt)
        response = self._vector_store.llm_query(prompt)
        result = evaluator.evaluate(
            response=response, contexts=[node.text for node in response_vector.nodes]
        )
        with logfire.span("Faithfulness"):
            logfire.info(
                "result",
                passing=result.passing,
                score=result.score,
                feedback=result.feedback,
                prompt=prompt,
                response=response,
                source=result.contexts[:1000],
            )
        return result

    @logfire.instrument
    def ragchecker(self, prompt: str, ground_truth: str):
        evaluator = RAGChecker(
            openai_api_key="lm-studio",
            extractor_api_base=self._llm.llm_base_url,
            checker_api_base=self._llm.llm_base_url,
            extractor_name=f"openai/{self._llm.model_name}",
            checker_name=f"openai/{self._llm.model_name}",
        )
        nodes = self._vector_store.query(prompt)
        query_response = self._vector_store.llm_query(prompt)

        wrapped = type("Response", (), {})()
        wrapped.response = query_response
        wrapped.source_nodes = [NodeWrapper(n) for n in nodes.nodes]

        result = response_to_rag_results(
            query=prompt,
            gt_answer=ground_truth,
            response_object=wrapped,
        )
        rag_results = RAGResults.from_dict({"results": [result]})

        metrics = evaluator.evaluate(
            rag_results,
            all_metrics,
        )
        return metrics


class NodeWrapper:
    def __init__(self, node: BaseNode):
        self.node = node
        self.id_ = node.id_
        self.node.text = node.get_content()

    def __repr__(self):
        return f"NodeWrapper({self.node})"
