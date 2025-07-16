# LLM Evaluation Server and UI

This repository contains a FastAPI backend and Gradio-based UI for evaluating large language model (LLM) outputs. The application uses:

- **FastAPI**: REST API and Gradio UI server  
- **Qdrant**: vector database for retrieval  
- **Jaeger (OpenTelemetry)**: tracing and metrics  
- **PostgreSQL**: persistent data storage  

All components are orchestrated via Docker Compose.

## Prerequisites

- Docker (v20.10 or newer)  
- Docker Compose (v2 plugin) or standalone `docker-compose`

## Quick Start with Docker

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. (Optional) Inspect and customize environment variables in `server/.env`:
   ```bash
   OTEL_EXPORTER_OTLP_ENDPOINT=http://otel:4318
   #OPENAI_API_KEY=<your_openai_key>
   OLLAMA_URL=http://host.docker.internal:1234/v1
   OLLAMA_MODEL=llama-3.2-3b-instruct
   OLLAMA_EMBEDDING_MODEL=text-embedding-bge-m3
   ```

3. Build and start all services in detached mode:
   ```bash
   docker compose up --build -d
   ```

4. Verify that containers are running:
   ```bash
   docker compose ps
   ```

## Stopping and Cleanup

To stop and remove containers, networks, and volumes created by Docker Compose:
```bash
docker compose down --volumes
```

## Accessing the Services

- **FastAPI / Gradio UI** (port `80`):
  - API Inference Endpoint:
    ```
    GET http://localhost/inference?prompt=Hello
    ```
  - Metrics Endpoint:
    ```
    GET http://localhost/metrics/{metric_type}?prompt=...&reference=...
    ```
  - Gradio UI:
    ```
    http://localhost/ui
    ```

- **Jaeger UI** (port `16686`):
  ```bash
  http://localhost:16686
  ```

- **Qdrant Vector DB** (port `6333`):
  ```bash
  http://localhost:6333
  ```

- **PostgreSQL** (port `5432`):
  - Host: `localhost`
  - User: `postgres`
  - Password: `postgres`
  - Database: `llm_eval`

## Viewing Logs

```bash
docker compose logs -f fastapi
docker compose logs -f otel
docker compose logs -f qdrant
docker compose logs -f db
```

## Notes

- If using OpenAI models, uncomment and set `OPENAI_API_KEY` in `server/.env`.  
- The Ollama service should be reachable at the URL specified by `OLLAMA_URL`.

## License

This project is licensed under the [MIT License](LICENSE).
