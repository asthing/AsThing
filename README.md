# AI Knowledge Workspace

An AI knowledge workspace where users will upload documents, organize them into knowledge bases, and ask questions answered from their sources.

The local Sprint 3 stack provides a Next.js app, FastAPI API, Celery worker, PostgreSQL, Redis, MinIO, and Qdrant.

## Prerequisites

- Docker Desktop with Docker Compose
- Ports `5432`, `6333`, `6379`, `8000`, `9000`, `9001`, and `3000` (or your chosen `WEB_PORT`) available
- About 5 GB of free Docker volume space for the local BAAI/bge-m3 model cache
- At least 2 GB more for the local Qwen2.5 3B Q4_K_M model cache, plus enough RAM for CPU inference

## Start the stack

```bash
docker compose up --build
```

No local `.env` file is required for development. To customize credentials or the web port, copy `.env.example` to `.env` and edit it. The checked-in defaults are development-only values. If port `3000` is already in use, set `WEB_PORT=3001` and open the web app on port `3001`.

Once the containers are healthy, open:

- Member web app: <http://localhost:3000>
- FastAPI health check: <http://localhost:8000/health>
- FastAPI OpenAPI docs: <http://localhost:8000/docs>
- MinIO console: <http://localhost:9001>
- Qdrant dashboard: <http://localhost:6333/dashboard>

## Knowledge bases

Open the **Knowledge Bases** navigation link to create and view the single local workspace's knowledge bases. Names are trimmed, must contain 1–100 characters, and are unique without regard to letter case. The API exposes `GET /knowledge-bases` and `POST /knowledge-bases`; the API container applies pending Alembic migrations before it starts.

## Documents

Open a knowledge base to upload one PDF, TXT, or DOCX source document at a time. Files may be up to 25 MiB. Each successful upload is stored in MinIO and recorded in PostgreSQL with an initial `Uploaded` status, then processed asynchronously by the Celery worker. The worker extracts text, persists deterministic chunks in PostgreSQL, generates BAAI/bge-m3 embeddings locally on CPU, and stores vectors in Qdrant. The first processing run downloads the model into the persistent `embedding_model_cache` volume.

The knowledge-base page lists documents newest first and polls for up to four minutes while any document is `Uploaded` or `Processing`. `Ready` means chunks and vectors are persisted for later retrieval. `Failed` shows a safe failure message and leaves no usable chunks or vectors. `GET /knowledge-bases/{knowledge_base_id}/documents` returns the document metadata and current public status.

The local worker uses one process and a prefetch multiplier of one so only one CPU-heavy BGE-M3 task is active at a time. It loads the immutable BAAI/bge-m3 revision `5617a9f61b028005a4858fdac845db406aefb181`; the named model-cache volume preserves that revision across restarts. `PROCESSING_LEASE_SECONDS` configures the durable processing lease and defaults to 300 seconds. The worker renews the lease during long phases, including the first model download and embedding generation. The API does not wait for Redis or Qdrant health to start. Its task-publication attempt has a one-second request deadline, so a temporarily unavailable Redis broker leaves a successfully uploaded document queued for recovery without allowing broker publication to hold the request longer than that bound.

## Find relevant sources

Within a knowledge base, submit one natural-language question to find up to five semantically relevant passages from `Ready` documents. Retrieval embeds the question locally with the pinned BGE-M3 model, asks Qdrant only for matching chunk identifiers in that knowledge base, then reads filenames, file types, and excerpts from PostgreSQL. It does not generate an answer. Documents that are `Uploaded`, `Processing`, or `Failed` are not searchable. If Qdrant or the local embedding model is unavailable, the API returns a safe retryable error while the API process itself remains available.

## Grounded answers

Before asking for an answer, initialize the local Ollama model explicitly:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:3b-instruct-q4_K_M
```

The API requires Ollama tag `qwen2.5:3b-instruct-q4_K_M` with manifest digest `sha256:357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`. It verifies this digest before each generation and rejects a changed or missing model. The named `ollama_model_cache` volume preserves the explicit download across restarts. Ollama is internal to Docker Compose, is limited to one active generation, and does not gate API startup. Its HTTP connection timeout is 2 seconds and generation read timeout is 180 seconds.

The knowledge-base question form offers **Get answer**. `POST /knowledge-bases/{knowledge_base_id}/answers` accepts only a question. The server retrieves up to five Ready chunks in that knowledge base and asks the local model to answer from those chunks. The response is either an answer with API-only source references, or an explicit `no_answer`. References are reserved for a later citation UI. No answers, prompts, or conversation history are stored.

Stop the stack with `docker compose down`. Add `--volumes` only when you intentionally want to delete local PostgreSQL, Redis, MinIO, Qdrant, and embedding-model-cache data.

## Repository layout

```text
apps/
  api/                 FastAPI core and AI service
  web/                 Next.js member application
packages/
  contracts/           Shared API schemas and generated types (later)
infra/
  docker/              Docker-specific support files
docs/
  adr/                 Architecture decision records
  runbooks/            Operational runbooks
  architecture.md      Current system overview
```

Nuxt admin, Django operations, Terraform, and Kubernetes are intentionally deferred according to the execution plan.

## Development checks

Run the API tests:

```bash
cd apps/api
python -m pip install -r requirements-dev.txt
pytest
```

Run the web checks:

```bash
cd apps/web
npm ci
npm run lint
npm run typecheck
npm run build
```
