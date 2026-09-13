# AI Knowledge Workspace

An AI knowledge workspace where users will upload documents, organize them into knowledge bases, and ask questions answered from their sources.

The repository is currently in Phase 1 (Foundation). This first vertical slice provides a local Next.js app, a FastAPI health endpoint, PostgreSQL, Redis, and MinIO.

## Prerequisites

- Docker Desktop with Docker Compose
- Ports `5432`, `6379`, `8000`, `9000`, `9001`, and `3000` (or your chosen `WEB_PORT`) available

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

## Knowledge bases

Open the **Knowledge Bases** navigation link to create and view the single local workspace's knowledge bases. Names are trimmed, must contain 1–100 characters, and are unique without regard to letter case. The API exposes `GET /knowledge-bases` and `POST /knowledge-bases`; the API container applies pending Alembic migrations before it starts.

Stop the stack with `docker compose down`. Add `--volumes` only when you intentionally want to delete local PostgreSQL, Redis, and MinIO data.

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
