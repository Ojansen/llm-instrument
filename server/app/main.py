import json
from typing import Annotated

import logfire
import os
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
import gradio as gr
from pydantic import BaseModel
from sqlalchemy import create_engine, ScalarResult
from sqlalchemy import select

from app.llm.openai import OpenAILLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logfire.configure(send_to_logfire=False, service_name="LLM instruments")
logfire.install_auto_tracing(modules=["/app"], min_duration=0.05)
logfire.instrument_fastapi(app, excluded_urls=["/ui", "/manifest.json"])
logfire.instrument_openai()
logfire.instrument_sqlalchemy(
    engine=create_engine("postgresql+psycopg://postgres:postgres@db:5432")
)
LlamaIndexInstrumentor().instrument()

# Import app so logfire can track methods
from app.llm.llama import Llama
from app.router.metrics import Metrics
from app.types import MetricType
from app.database.db import Database
from app.database.models import Project, Session, Prompt, PromptResponse, TestCase
from app.utils.interface import Interface

# Setup llama-index
llm = OpenAILLM()
metrics_instance = Metrics(llm=llm)
Settings.llm = llm.agent
Settings.embed_model = llm.embedding_model

# Disable Gradio interface
# interface = Interface(llm=llm)
# gr.mount_gradio_app(app, interface.render(), path="/ui")

db = Database()


@app.get("/inference")
async def inference(prompt: str):
    output = await llm.inference(prompt)
    return {"output": output}


@app.get("/projects")
def get_projects():
    stmt = select(Project).order_by(Project.id)
    result: ScalarResult[Project] = db.Session.scalars(stmt)
    return [project for project in result]


@app.get("/projects/{project_id}")
def get_project(project_id: int):
    stmt = select(Project).where(Project.id == project_id)
    result: ScalarResult = db.Session.scalars(stmt)
    return result.first()


@app.get("/projects/{project_id}/sessions")
def get_projects_sessions(project_id: int):
    stmt = select(Session).where(Session.project.has(Project.id == project_id))
    result: ScalarResult[Session] = db.Session.scalars(stmt)
    return [session for session in result]


@app.get("/{session_id}/evals")
def get_session(session_id: int):
    result = (
        db.Session.query(PromptResponse, Prompt, TestCase)
        .join(Prompt)
        .join(TestCase)
        .filter(Prompt.session_id == session_id)
        .all()
    )
    return [
        {
            "system_prompt": prompt.system_prompt,
            "question": testcase.question,
            "answer": testcase.expected_answer,
            "model_data": {
                "name": response.model_name,
                "temperature": response.temperature,
                "latency_ms": response.latency_ms,
                "token_usage": response.token_usage,
            },
            "metrics": {
                "correctness": response.correctness_score,
                "faithfulness": response.faithfulness,
                "feedback": response.feedback,
            },
        }
        for response, prompt, testcase in result
    ]


class EvalFormData(BaseModel):
    session_id: int
    system_prompt: str
    question: str
    reference: str


@app.post("/{session_id}/eval")
async def create_eval(data: EvalFormData):
    print(data)
    llm.system_prompt = (
        data.system_prompt if data.system_prompt != "" else llm.system_prompt
    )
    cosine = await metrics_instance.cosine_similarity(
        prompt=data.question, reference=data.reference
    )
    faithfulness = await metrics_instance.faithfulness(prompt=data.question)
    return {
        "cosine": cosine.feedback,
        "faithfulness": faithfulness.feedback,
    }


@app.get("/metrics/{metric_type}")
async def metrics(metric_type: MetricType, prompt: str, reference="", ground_truth=""):
    match metric_type:
        case MetricType.similarity:
            return await metrics_instance.cosine_similarity(
                prompt=prompt,
                reference=reference,
            )
        case MetricType.correctness:
            return await metrics_instance.correctness(
                prompt=prompt,
                reference=reference,
            )
        case MetricType.faithfulness:
            return await metrics_instance.faithfulness(
                prompt=prompt,
            )
        case MetricType.ragchecker:
            return metrics_instance.ragchecker(
                prompt=prompt,
                ground_truth=ground_truth,
            )
        case _:
            raise Exception(f"Unknown metric type: {metric_type}")
