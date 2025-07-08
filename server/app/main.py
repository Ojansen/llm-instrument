import logfire
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.evaluation import SemanticSimilarityEvaluator
import gradio as gr

from app.router.metrics import Metrics
from app.llm.lmstudio import LmStudio
from app.router.inference import Inference

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
logfire.instrument_fastapi(app)
logfire.instrument_pydantic_ai()

llm = LmStudio(system_prompt="")
agent = llm.agent()


@app.get("/inference")
def inference(prompt: str):
    output = Inference(agent=agent).run(prompt)
    return {"output": output}


@app.get("/similarity")
def similarity(prompt: str, reference: str):
    metrics = Metrics(agent=agent)
    result = metrics.cosine_similarity(prompt=prompt, reference=reference)

    return result


io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, io, path="/ui")
