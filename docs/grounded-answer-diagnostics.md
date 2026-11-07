# Grounded-answer failure diagnosis (US 6.1)

This is opt-in measurement infrastructure. Production answer/retrieval code, prompts,
models, the v1 manifest, and established evaluation reports are unchanged. v1 is a
fixed, previously inspected regression benchmark, not unseen validation data.

From the repository root, run once:

```powershell
.\scripts\run-grounded-answer-diagnostic.ps1
```

Prerequisites and dedicated cache prewarming are the same as the README's
grounded-answer evaluation instructions. Diagnostic execution never downloads models.
Missing dedicated caches or pinned identities stop before seeding or generation with:
`Diagnostic preflight failed. Run the documented evaluation prewarm commands.`

The script uses a fresh `asthing-diagnostic-...` project, the existing port-free
evaluation override, and a diagnostic-only override. It checks the effective Compose
configuration, mounts fictional fixtures read-only, and disables evaluator container
logging. Normal HTTP uploads and the real worker prepare Ready fixtures. Existing
isolated seeding prepares Uploaded/Failed fixtures; only the 12 expected-answer cases
are diagnosed. PostgreSQL/MinIO/Qdrant/Redis data volumes are disposable; only the two
dedicated evaluation model-cache volumes persist. The script's `finally` block removes
only its unique project. A forced host termination may bypass `finally`; use the exact
printed project name with all three Compose files to remove that project, never the
default stack or dedicated cache volumes.

Native Docker build/startup/cleanup output is suppressed by the orchestration script;
failures produce fixed safe messages. Operational run labels and the output location
are printed so the unique disposable project and its artifacts can be identified.

The diagnostic calls the unchanged `AnswerService`, `SourceRetrievalService`, and
`LocalAnswerModel` inside its isolated process. It observes real PostgreSQL hydration,
Qdrant retrieval, the actual top-one model context, and one generation per case that
has evidence. It does not duplicate the answer implementation or generate a second
answer for diagnosis. Unlike US 5.2, answer diagnosis occurs in-process rather than
through the public HTTP answer route because the safe public 503 deliberately hides
the validation-failure distinction. Fixture preparation still uses real HTTP routes.

## Output and interpretation

```powershell
Get-ChildItem .data\grounded-answer-diagnostics -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content .data\grounded-answer-diagnostics\<run-id>\decision-summary.md
Get-Content .data\grounded-answer-diagnostics\<run-id>\diagnostic.json
```

These ignored artifacts contain case IDs, fixed classifications/review categories,
booleans, and counts only. They never contain questions, generated answers, excerpts,
filenames, full source text, prompts, runtime/database IDs, addresses, vectors, or
exception details. Established US 5.2 reports are not touched.

Classification precedence is:

1. Execution failure: no quality classification. Continue later cases; mark the run
   incomplete. Cache/preflight failures abort globally. A successful run covers exactly
   12 distinct expected-answer case IDs.
2. Reference-specific invalid output, then other invalid structured output.
3. A valid answer/reference response that passes the unchanged canonical matcher is
   `matched`.
4. Empty retrieval or confirmed human review establishes `retrieval_miss`.
5. Empty actual context or confirmed human review establishes
   `answer_context_missing_fact`.
6. `no_answer` with the selected expected source's canonical fact in actual context,
   or confirmed human evidence support, is `false_abstention`.
7. A generated answer with documented human confirmation of omission is
   `canonical_fact_omission`.
8. All other successfully observed cases remain `semantic_match_uncertain`.

Canonical-string presence is recorded separately for retrieved chunks and actual
context. Absence alone establishes neither lack of semantic evidence nor an incorrect
answer. The classifier never automatically upgrades a paraphrase miss to a model
failure. Human review can be represented only by a case ID and one fixed decision:
`retrieved_evidence_missing`, `context_evidence_missing`, `context_support_confirmed`,
or `answer_fact_omitted`. The initial run leaves these null; no human decisions are
fabricated. The pure `safe_result`/`classify` functions support those decisions for
subsequent reviewed diagnostic artifacts without changing the benchmark.

Raw inspection requires an explicit flag and an interactive terminal, but **is disabled
in this implementation even with a TTY**. Docker/host transcripts cannot guarantee an
unrecorded console path. `--inspect-console` fails safely before seeding. No raw content
is printed or logged by the diagnostic. Third-party Python logging is disabled in its
entry point; the diagnostic Docker logging driver is `none`.

## Decision and next backlog item

The report recommends exactly one hypothesis. Uncertainty or incomplete execution
takes priority over tuning. Otherwise it chooses a single observed cause (context gap,
retrieval gap, invalid references, invalid structure, confirmed abstention, or confirmed
omission), never a collection of tuning ideas. Case IDs support the recommendation.

When uncertainty dominates, the proposed next item is **Resolve grounded-answer scoring
uncertainty before tuning**. Hypothesis: canonical sentence matching may reject concise
supported answers. Minimal scope: reviewed diagnostic-only case decisions and a safe
human-review procedure, preserving the original v1 score alongside the reviewed
assessment. No production behavior or canonical matcher changes.

Proposed acceptance gates, to approve before that item's implementation:

- Cover all 12 answerable cases (6 calibration and 6 previously inspected holdout).
- Resolve every currently uncertain case to a documented fixed review decision or
  explicitly retain uncertainty; 0 unresolved cases is required before proposing tuning.
- 0 automatically inferred semantic successes, 0 fixture/holdout edits, and 0 raw-content
  artifact/log leaks; canonical v1 results remain reproducible and separately labeled.
- Keep production regression tests passing. A later production candidate must retain
  retrieval hit@5 1.0, valid-reference rate 1.0, 0 isolation failures, 0 unsupported
  answers across all 8 negative cases, and meet preregistered holdout completion gates
  (at least 3/4 direct and 4/6 overall; direct false abstentions 0).

This story does not implement that next item or claim independent generalization or
quality improvement. A 12-case diagnostic run alone does not remeasure the eight
negative-case safety gates.

## Reviewed isolated run

The single US 6.1 run on 2026-10-14 used project
`asthing-diagnostic-20261014-220946-cd680f1c`. It completed all 12 distinct
expected-answer cases with no execution failures. All 12 had the expected source and
the canonical string in both retrieved chunks and actual model context, and each
invoked generation once. Results:

| Group | False abstention | Semantic match uncertain |
| --- | ---: | ---: |
| Calibration | 3 | 3 |
| Previously inspected holdout | 2 | 4 |
| Direct evidence | 2 | 6 |
| Paraphrased evidence | 3 | 1 |

All other primary classification counts were zero. There were no observed invalid
model outputs, invalid references, retrieval misses, or confirmed context omissions.
The seven uncertainties are `direct-evidence-001`, `direct-evidence-002`,
`direct-evidence-004`, `direct-evidence-005`, `direct-evidence-006`,
`direct-evidence-008`, and `paraphrased-evidence-004`. Their valid answers fail the
unchanged matcher; their semantic correctness has not been established.

The reviewed decision is exactly one next hypothesis: **resolve measurement uncertainty
first** using the backlog item above. This does not assert that all seven answers are
correct or erase the five confirmed abstentions. The diagnostic summary and safe JSON
are retained under `.data/grounded-answer-diagnostics/20261014-220946-cd680f1c/`.
The run verified offline pinned caches, zero published ports, read-only fixtures, and
disabled diagnostic logging. Its unique containers, network, and four disposable data
volumes were removed; both dedicated evaluation caches were preserved.

## Deterministic verification

```powershell
cd apps/api
python -m pytest tests/test_grounded_answer_diagnostic.py
python -m pytest
ruff check .
```

The tests use fake retrieval/HTTP clients, never real models or infrastructure. They
exercise production answer/model validation with those fakes, classification
precedence, one-generation behavior, execution failure continuation, safe report
allowlisting, immutable fixtures, console gating, and sentinel-based privacy checks.
