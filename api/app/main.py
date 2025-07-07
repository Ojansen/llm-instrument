import logfire
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from logging import basicConfig, getLogger

from app.evals import dataset, guess_city
from app.agent import agent

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

otel_exp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
if not otel_exp_endpoint:
    print("OTEL_EXPORTER_OTLP_ENDPOINT not set")


os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = otel_exp_endpoint
logfire.configure(send_to_logfire=False)
logfire.instrument_fastapi(app)
logfire.instrument_pydantic_ai()


@app.get("/inference")
def inference(prompt: str):
    result_sync = agent.run_sync(prompt)
    return {"output": result_sync.output}


@app.get("/similarity")
def similarity(prompt: str, reference: str):
    evaluator = SemanticSimilarityEvaluator()
    result_sync = agent.run_sync(prompt)
    result = evaluator.evaluate(
        response=result_sync.output,
        reference=reference
    )
    with logfire.span('Cosine Similarity'):
        logfire.span(result.feedback)

    return {"score": result.score, "passing": result.passing, "output": result_sync.output}
