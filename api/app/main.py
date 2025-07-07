import logfire
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
def inference(query: str):
    result_sync = agent.run_sync(query)
    print(result_sync)
    return {"output": result_sync.output}


@app.get("/similarity")
def similarity(query: str):
    result_sync = agent.run_sync(query)