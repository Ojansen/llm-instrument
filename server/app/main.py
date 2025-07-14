import logfire
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
import gradio as gr
from sqlalchemy import create_engine

app = FastAPI()


otel_exp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
if not otel_exp_endpoint:
    print("OTEL_EXPORTER_OTLP_ENDPOINT not set")


os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_exp_endpoint
logfire.configure(send_to_logfire=False, service_name="LLM instruments")
logfire.install_auto_tracing(modules=["/app"], min_duration=0.01)
logfire.instrument_fastapi(app, excluded_urls=["/ui", "/manifest.json"])
logfire.instrument_openai()
logfire.instrument_sqlalchemy(
    engine=create_engine("postgresql+psycopg://postgres:postgres@db:5432")
)
LlamaIndexInstrumentor().instrument()


from app.llm.llama import Llama
from app.router.metrics import Metrics
from app.types import MetricType
from app.utils.interface import Interface


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


@app.get("/inference")
def inference(prompt: str):
    output = llm.inference(prompt)
    return {"output": output}


@app.get("/metrics/{metric_type}")
def metrics(metric_type: MetricType, prompt: str, reference=""):
    metrics_instance = Metrics(llm=llm)

    if metric_type == MetricType.similarity:
        return metrics_instance.cosine_similarity(prompt=prompt, reference=reference)

    if metric_type == MetricType.correctness:
        return metrics_instance.correctness(prompt=prompt, reference=reference)

    if metric_type == MetricType.faithfulness:
        return metrics_instance.faithfulness(prompt=prompt)

    return "Invalid metric type"


llm = Llama(
    system_prompt="You are a helpful assistant and should answer the following questions:"
)
Settings.llm = llm.agent
Settings.embed_model = llm.embedding_model
interface = Interface(llm=llm)
app = gr.mount_gradio_app(app, interface.render(), path="/ui")
