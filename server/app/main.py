import logfire
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from app.llm.llama import Llama
from app.router.metrics import Metrics
from app.types import MetricType
from app.utils.interface import Interface

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


os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_exp_endpoint
logfire.configure(send_to_logfire=False)
logfire.instrument_fastapi(app, excluded_urls=["/ui", "/manifest.json"])
logfire.instrument_pydantic_ai()
logfire.instrument_openai()

llm = Llama(
    system_prompt="You are a helpful assistant and should answer the following questions:"
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


interface = Interface(llm=llm)
app = gr.mount_gradio_app(app, interface.render(), path="/ui")
