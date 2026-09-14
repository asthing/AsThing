# AI Knowledge Workspace — Execution Plan

**Timeline:** 07 Sep 2026 – 28 Feb 2027  
**Primary goal through 31 Dec 2026:** TOEIC 550 → 800.  
**Project allocation through December:** 6–8 hours/week.  
**Rule:** Deliver a working vertical slice before adding another technology.

---

## 1. Product definition

Build an AI knowledge workspace where a user can upload documents, organize them into a knowledge base, and ask questions answered only from the uploaded sources.

### Users

- **Member:** uploads documents, searches, chats with a knowledge base, sees cited sources.
- **Admin:** manages users, documents, processing failures, prompt versions, and answer feedback.

### Success criteria

By February 2027, demonstrate that you can:

1. Build and test a full-stack product end-to-end.
2. Build a reliable RAG workflow with citations and evaluation data.
3. Deploy, monitor, and troubleshoot it on AWS using infrastructure as code.
4. Explain design trade-offs clearly in English.

---

## 2. Architecture and technology decisions

| Concern | Choice | Why now |
|---|---|---|
| Member web app | Next.js + TypeScript | Main user experience and server/client React patterns. |
| Admin web app | Nuxt + TypeScript | Separate admin workflow; introduce after the core API works. |
| Core API / AI | FastAPI | Async-friendly APIs, typed contracts, streaming AI responses. |
| Internal operations | Django + Django Admin | Learn Django through a focused internal moderation/configuration tool. |
| Source of truth | PostgreSQL + Qdrant | PostgreSQL stores durable metadata and chunks; Qdrant stores vectors. |
| Raw extracted document data | MongoDB | Flexible parser output and versioned document metadata. |
| Async jobs | Celery + Redis locally; SQS on AWS | File processing and embeddings must not block HTTP requests. |
| Object storage | MinIO locally; S3 on AWS | Same file-storage contract in local and cloud environments. |
| Events | Kafka | Add only after the MVP: document/index/feedback audit events. |
| AWS-native key/value data | DynamoDB | Add only for idempotency/audit use case; do not duplicate PostgreSQL data. |
| Deployment | Docker → ECS Fargate → Kubernetes | First prove deployment on ECS; learn K8s only after the product is stable. |
| Infrastructure | Terraform | Reproducible AWS resources and reviewable cloud changes. |
| CI/CD | GitHub Actions | Test, build image, push to ECR, deploy staging. |

### Non-negotiable scope control

- Do **not** start Kafka, DynamoDB, Kubernetes, or EKS before the RAG MVP works.
- PostgreSQL is the source of truth. Do not use multiple databases for the same data.
- Do not fine-tune a model in this project. Prioritize RAG quality, evaluation, latency, and cost.
- One feature is complete only when it has a test, error handling, and a short README update.

---

## 3. MVP scope (build this first)

### Must have

1. Email/password authentication and a protected workspace.
2. Create a knowledge base.
3. Upload PDF/TXT/DOCX files.
4. Background processing: extract text, chunk, embed, index.
5. Display document status: uploaded, processing, ready, failed.
6. Ask a question and stream an answer.
7. Show source citations for each answer.
8. Let users mark an answer helpful/not helpful.

### Explicitly later

- Multi-tenant billing.
- Complex visual editor.
- Team invitations.
- Kafka, DynamoDB, Kubernetes, full AWS production setup.
- Fine-tuning, agents, multi-modal processing.

---

## 4. Repository layout

```text
ai-knowledge-workspace/
├── apps/
│   ├── web/                 # Next.js member application
│   ├── admin/               # Nuxt admin application
│   ├── api/                 # FastAPI core and AI service
│   └── ops/                 # Django + Django Admin
├── packages/
│   └── contracts/           # API schemas/types shared where practical
├── infra/
│   ├── docker/
│   ├── terraform/
│   └── k8s/
├── docs/
│   ├── adr/
│   ├── runbooks/
│   └── architecture.md
├── docker-compose.yml
├── README.md
└── .github/workflows/
```

---

## 5. Delivery roadmap

### Phase 1 — Foundation (07 Sep – 04 Oct)

- Create repository, issue board, branch protection, PR template.
- Docker Compose: web, API, PostgreSQL, Redis, MinIO.
- FastAPI health check, OpenAPI, SQLAlchemy/Alembic migration.
- Next.js login shell and authenticated API call.
- First vertical slice: create/list a knowledge base.

**Definition of done:** `docker compose up` starts the stack; a user can sign in and create a knowledge base.

### Phase 2 — Documents and operations (05 Oct – 01 Nov)

- Upload a document to MinIO/S3-compatible storage.
- Add document record/status in PostgreSQL.
- Create Nuxt admin list/detail pages.
- Create minimal Django Admin for internal document moderation/config.
- Add FastAPI unit/integration tests and one Playwright happy-path test.

**Definition of done:** upload is safe, observable, and manageable from an admin screen.

### Phase 3 — RAG MVP (02 Nov – 06 Dec)

- Celery job: extract → normalize → chunk → embed → store vectors.
- Search relevant chunks with Qdrant.
- Streaming RAG answer with source citations.
- Store answer feedback and logs.
- Build a 30–50 question evaluation set with expected source references.

**Definition of done:** every answer is traceable to retrieved source chunks and can be evaluated.

### Phase 4 — Stabilize for TOEIC period (07–31 Dec)

- Fix defects; improve README; write architecture decisions.
- Do not add new infrastructure or AI features.
- Capture one short English demo per week.

### Phase 5 — AWS and CI/CD (Jan 2027)

- Terraform: VPC, IAM, S3, RDS, ECR, ECS Fargate, SQS, CloudWatch.
- GitHub Actions: lint → test → build → push image → staging deploy.
- Replace local job broker with SQS where appropriate.
- Structured logging, health checks, monitoring, rollback runbook.

### Phase 6 — Senior-level system design (Feb 2027)

- Kafka for product/audit events only if it improves a real flow.
- DynamoDB for idempotency/audit only if PostgreSQL is demonstrably unsuitable.
- Kubernetes locally; EKS only if budget and a concrete deployment reason exist.
- Load test, security review, cost analysis, final English demo and case study.

---

## 6. Weekly workflow

Until 31 Dec, follow this cadence after TOEIC work:

| Day | Project block | Outcome |
|---|---:|---|
| Tuesday | 60 min | Implement one small backend or UI task. |
| Thursday | 60 min | Continue; commit and write a short English technical note. |
| Saturday | 2 hours | Deep work: complete the vertical slice. |
| Sunday | 3 hours | Tests, bug fixes, documentation, and next-week plan. |

### One-week rule

Each week must end with exactly one visible outcome:

- a merged feature,
- a test suite improvement,
- an architecture decision record,
- a deployment improvement, or
- a demo recording.

Do not end a week with only tutorials watched.

---

## 7. Working with Codex efficiently

### Before asking Codex

1. Open or create a GitHub issue with a small, testable outcome.
2. State the relevant files, acceptance criteria, and constraints.
3. Ask for a plan first when the change affects more than 3 files.
4. Keep changes in a focused branch/PR.

### Prompt template: implement a small feature

```text
Work in this repository. First inspect the relevant existing code and summarize
the current flow in 5 bullets. Then propose the smallest implementation plan.

Task: <one concrete task>
Acceptance criteria:
- <criterion 1>
- <criterion 2>
- Add/update tests.
- Do not change unrelated files.

After implementation, run the relevant checks and report: files changed,
commands run, results, and any remaining risk.
```

### Prompt template: learn while building

```text
I am learning <topic> through this project. Explain only the concepts needed
for this task, then guide me to implement it in small steps. Stop after each
step and ask me to run/verify it before continuing. Prefer the project’s
existing conventions over a new abstraction.
```

### Prompt template: debug

```text
Diagnose this issue without changing code first.

Expected behavior: <expected>
Actual behavior: <actual>
Reproduction steps: <steps>
Logs/error: <paste>

Inspect the most likely files/configuration. Give the root-cause hypothesis,
evidence, and the smallest safe fix. Implement only after confirming the plan.
```

### Prompt template: code review

```text
Review this change as a senior full-stack/AI engineer. Check correctness,
security, test coverage, performance, API design, observability, and scope.
List findings by severity. Do not rewrite code unless I request a fix.
```

---

## 8. Quality gates

Run before every PR:

```text
- formatter/linter passes
- type checking passes
- API tests pass
- relevant UI tests pass
- Docker build succeeds
- migration works on an empty local database
```

For AI features, also record:

```text
- prompt version
- retrieved sources
- latency
- token/cost estimate
- user feedback
- evaluation result
```

---

## 9. Portfolio deliverables by February

- Public README with local setup and architecture diagram.
- Three ADRs: data-store choices, queue/event choices, ECS vs Kubernetes.
- Demo environment or video walkthrough.
- 5–8 minute English technical demo.
- Two written case studies: RAG quality/evaluation and cloud deployment/operations.
- Incident/rollback runbook and basic load-test results.

## 10. First action

Create the repository and complete only this issue:

```text
Title: Bootstrap local development environment

Acceptance criteria:
- Monorepo directories exist.
- Docker Compose starts PostgreSQL, Redis, MinIO, FastAPI, and Next.js.
- GET /health returns 200 from FastAPI.
- README documents one command to start the stack.
```

Do not create Nuxt, Django, Kafka, DynamoDB, AWS, Terraform, or Kubernetes yet.
