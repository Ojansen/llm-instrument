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

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://otel-tui:4318'
logfire.configure(send_to_logfire=False)
logfire.instrument_fastapi(app)


@app.get("/hello")
def hello(query: str):
    # report = dataset.evaluate_sync(guess_city)
    result_sync = agent.run_sync(query)
    return {"message": result_sync.output}
