# Permanent document deletion

One confirmed action deletes one document in its selected knowledge base. Uploaded,
Ready, and Failed documents are eligible. Processing documents are rejected by the
backend, including expired processing leases; this feature does not cancel processing.
Filenames are displayed as text and are never used as storage identifiers.

## API and UI

`DELETE /knowledge-bases/{knowledge_base_id}/documents/{document_id}` has no body.

- `204` with an empty body: original object and document vectors are removed, and
  PostgreSQL chunk/job/document deletion has committed.
- `404`: knowledge base missing, or document absent/outside the selected knowledge base.
  Repeating deletion of an absent resource does no external work.
- `409`: `This document is processing and cannot be deleted. Wait for processing to finish.`
- `503` before ownership: `Document deletion is unavailable. Please try again.`
- `503` after ownership: `Deletion is incomplete. The document is excluded from searches. Retry deletion.`
- `422`: existing invalid UUID/path validation.

Errors use the existing JSON `detail` shape. Known dependency/connection/lock failures
are safe 503 responses; unexpected programming errors retain normal server handling.
No object keys, vector identifiers, source contents, credentials, or raw dependency
errors are returned. Public processing statuses remain Uploaded/Processing/Ready/Failed.

Document responses add `deletion_pending`, default false. The UI confirms the filename
and permanent removal, disables duplicate submissions, and keeps the row until 204.
On partial failure/reload, the row shows Deletion incomplete and Retry deletion instead
of a searchable Ready label. Pending deletion participates in the existing two-second,
120-attempt polling lifecycle, so worker recovery completion disappears without refresh.
Retry/reload restarts that bounded cycle. Successful deletion cancels older list requests;
a knowledge-base-scoped set of confirmed deleted IDs also filters late list snapshots.
Remaining order and the existing empty state are preserved.

## Durable intent and coordination

Migration `20261107_01` adds nullable `documents.deletion_requested_at`; existing rows
start with no deletion intent. It changes no status constraint or existing file data.
Apply it before starting the updated worker. Downgrade refuses while any deletion
intent remains, so a rollback cannot silently discard cleanup obligations. PostgreSQL
downgrade acquires `ACCESS EXCLUSIVE` on `documents` **before** checking intent and
holds it through column removal in the same Alembic transaction. Concurrent intent
updates cannot pass between the check and DDL. Use a maintenance window: a writer
blocked behind a successful downgrade fails once the column is gone.

Migration `20261108_01` adds nullable `vector_write_token` and `vector_write_started_at`
to processing jobs. Existing queued/settled successful records start with neither.
Pre-upgrade Processing/CleanupPending jobs and recorded vector/cleanup failures are
conservatively backfilled with the job UUID as token: their old code had no durable
acknowledgement journal. This may quarantine work that never issued a vector write.
Its downgrade also locks the job table and refuses unresolved write tokens.
Apply both migrations before restarting updated API/worker processes; do not mix old
workers, which do not record write intent, with the new deletion service.

All participating operations use processing-job-first, document-second row locking:

1. DELETE locks the processing job (when present), then the document. It validates the
   knowledge base/status and commits deletion intent before contacting external stores.
2. Processing claims already update/lock the job first. Their document update now also
   requires no deletion intent. If a claim commits first, DELETE sees Processing and
   returns 409. If deletion commits first, a claim rolls back without processing.
3. Chunk persistence and vector upsert lock those same rows and revalidate job ID,
   current attempt, document attempt/status, and absence of deletion intent **inside the
   locked section**. Chunk writes retain locks through commit. Before dispatching an
   upsert, the worker commits a fresh write token on the current job, then reacquires
   job/document locks and validates that exact token and attempt. It retains those
   locks through the synchronous upsert and clears the token only after a `completed`
   acknowledgement and committed database update. A waiting stale writer cannot write.
   Locks and lease heartbeats alone are **not** remote write fences.
4. Cleanup locks the job and document again and keeps them locked through completed
   Qdrant deletion, strict MinIO deletion, and final PostgreSQL commit. Other DELETE and
   recovery calls serialize on these rows and never use previously cached object keys
   after acquiring the locks. Existing legacy documents without a job use their document
   row lock; processing claims cannot manufacture a missing job.

Deletion lock waits are bounded at five seconds. Strict MinIO deletion uses a separate
client with a two-second connection timeout, five-second read timeout, and one attempt;
upload/download/compensation settings and best-effort upload compensation stay unchanged.
Qdrant deletion uses `wait=True` and requires a completed operation result. Missing
objects, buckets, collections, or document vectors are already clean. Authentication,
connection, and other infrastructure failures are not treated as missing resources.

The document, unique original object key, chunks/point IDs, and processing job are
retained until both external cleanups succeed and the final transaction commits. If
that commit fails, database changes roll back while durable intent remains. Repeating
the acknowledged external deletions is safe. Recovery never starts deletion for an
unmarked document and delayed processing tasks cannot claim marked or missing records.

Both Ready-document detection and PostgreSQL chunk hydration exclude deletion intent.
Thus even a stale Qdrant hit cannot supply new retrieval evidence. Requests already in
flight may have captured evidence before deletion acceptance; their answer and displayed
citations are not revoked. Browser cancellation also cannot undo committed intent.

## Explicit bounded recovery

Restore the unavailable dependency, then use Retry deletion or invoke existing worker
recovery:

```powershell
docker compose exec worker celery -A app.celery_app.celery_app call documents.recover
```

For a smaller pass:

```powershell
docker compose exec worker celery -A app.celery_app.celery_app call documents.recover --args '[20]'
```

Each recovery pass handles at most 100 processing candidates and 100 deletion candidates
(or the smaller supplied batch size). Processing recovery excludes deletion intent;
deletion recovery publishes `documents.delete` outside the discovery transaction.
Redis failure retains intent. Dedicated deletion tasks are idempotent and serialize
with API cleanup. There is no new periodic scheduler: restart retains work, but an
explicit recovery pass or UI Retry is required to resume it.

PostgreSQL/MinIO/Qdrant do not share a transaction. Do not remove the retained database
row to "fix" a cleanup error; it is the durable obligation and remaining identifiers.
Administrative direct writes to MinIO/Qdrant bypass application coordination and are
outside this feature's guarantees.

## Unacknowledged vector writes: deliberate fail-closed recovery

The pinned client is `qdrant-client==1.14.2`; the server is `qdrant/qdrant:v1.15.5`.
The installed client's `upsert` accepts `wait` and `ordering` and returns a server-assigned
`operation_id`/status; it has no caller-assigned operation token or operation-status
lookup method. The [v1.15 upsert contract](https://api.qdrant.tech/v-1-15-x/api-reference/points/upsert-points)
defines waiting for completion and write ordering, but does not supply a cancellation
or conditional fencing primitive for a request whose acknowledgement was lost. We do
not assume that `strong` ordering orders a still-unobserved/delayed request before a
later deletion, or that it settles an unidentified operation after connection loss.

Consequently an upsert timeout, disconnect, or worker loss retains the token, timestamp,
current attempt, document/object identifiers, and persisted chunks/point identifiers.
Known processing failure moves the job to internal `CleanupPending` and keeps the
document public `Processing`. Recovery of an expired Processing lease with a token
quarantines that **same attempt** in CleanupPending; it never starts another upsert.
Cleanup may remove currently visible vectors repeatedly, but cannot remove chunks or
finalize Failed while the token remains. Ready completion also rejects a pending token.

Processing documents still return 409 on DELETE. If deletion intent already exists on
an otherwise eligible record, unresolved write intent makes cleanup return the existing
safe incomplete 503; the marker and all identifiers remain. Retrieval excludes pending
deletion and Processing documents regardless of any stale/late vector hits.

Crash boundaries deliberately favor safety:

- Before intent commits: no RPC is sent.
- After intent commit, before sending: a crash conservatively leaves unresolved intent.
- During RPC, after timeout, or after worker loss: remote completion is unknown; retain intent.
- After remote completion but before the acknowledgement-clearing transaction commits:
  retain intent, even if Qdrant actually completed the operation.
- After acknowledgement clearing commits: that write is settled; normal completion or
  cleanup/deletion may proceed under the existing locks.

**Limitation:** this pinned interface cannot safely settle an unidentified lost-ACK
operation automatically. Explicit recovery/Retry is idempotent but may remain pending
indefinitely. Restart, a grace period, an empty vector query, and successful deletion
are not settlement proofs and never clear the token. There is intentionally no force-clear
API, timeout-based override, or automatic re-upsert. A future administrative settlement
mechanism needs separately verified remote completion or effective fencing before it
can clear this obligation. Do not manually delete these records/tokens to obtain 204.
This does not reconstruct an unknowable history of already-completed pre-upgrade
attempts; rollout must stop old workers and audit any suspected older unrecorded writes.

## Verification

Normal tests use SQLite and fake external stores. Real lock tests additionally require
`DELETION_TEST_DATABASE_URL` pointing to **the disposable test project's PostgreSQL**:

```powershell
python -m pytest tests/test_document_deletion.py
python -m pytest tests/test_vector_write_safety.py
python -m pytest tests/test_document_deletion_postgres.py -v
python -m pytest
ruff check .
```

The PostgreSQL tests create/drop unique test schemas, use concurrent sessions, and
require `pg_stat_activity.wait_event_type = 'Lock'` before releasing competing operations.
They cover both claim/delete orderings, blocked stale chunk/vector writers, an in-flight
acknowledged vector write, and concurrent DELETE/recovery cleanup. Without the explicit
disposable URL they are skipped, not represented as SQLite concurrency evidence.
They also verify the downgrade check/drop lock and refusal of existing pending intent.
Deterministic fault tests cover lost acknowledgements, worker loss, late completion
during cleanup, repeated recovery, and preservation of independent documents.

Frontend Node tests run from the bracket-named route directory (avoids shell/test-runner
glob ambiguity), followed by the usual checks:

```powershell
Set-Location -LiteralPath 'apps/web/app/knowledge-bases/[knowledgeBaseId]'
node --experimental-strip-types --test *.test.ts
```

Use literal-path directory handling in PowerShell if needed. Run web lint, typecheck,
and production build from `apps/web`.

For the real-stack smoke test use a unique Compose project and the existing evaluation
override for disposable database/object/vector/broker data and dedicated offline caches.
An ephemeral additional override publishes only disposable loopback web/API ports.
Use fictional files, including independent files with the same name in the same and
another KB. Cover Uploaded/Ready/Failed deletion, active Processing rejection, strict
store removal, MinIO outage, retained intent, restart, UI Retry and worker recovery,
controlled stale vector hits, duplicate task delivery, browser confirmation/cancellation,
progress/pending state, automatic row removal and the empty state. Remove only that
unique project's containers/network/data volumes, preserving external model caches.
