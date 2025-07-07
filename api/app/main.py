import logfire
import os
from fastapi import FastAPI

app = FastAPI()

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://otel-tui:4318'
logfire.configure(send_to_logfire=False)
logfire.instrument_fastapi(app)


@app.get("/hello")
async def hello(name: str):
    return {"message": f"hello {name}"}
