from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

import app.grounded_answer_evaluation as evaluation
from app.document_models import Document, DocumentProcessingJob
from app.grounded_answer_evaluation import (
    EvaluationCase,
    FixtureDocument,
    FixtureKnowledgeBase,
    build_report,
    classify_case_result,
    load_manifest,
    preflight_model_caches,
    run_evaluation,
    write_reports,
)
from app.knowledge_bases import KnowledgeBase

READY_KNOWLEDGE_BASE = FixtureKnowledgeBase(
    "engineering-handbook",
    (
        FixtureDocument(
            "engineering-handbook.txt", "knowledge_bases/engineering-handbook.txt", "Ready"
        ),
    ),
)


def answer_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="direct-evidence-001",
        split="calibration",
        category="direct_evidence",
        knowledge_base="engineering-handbook",
        question="What must happen before a release?",
        expected_outcome="answer",
        expected_source_filename="engineering-handbook.txt",
        expected_canonical_fact="automated checks must pass before a release",
        outside_matching_evidence=None,
        unavailable_matching_evidence=None,
    )


def no_answer_case(*, category: str = "no_supporting_evidence") -> EvaluationCase:
    return EvaluationCase(
        case_id="no-evidence-001",
        split="calibration",
        category=category,
        knowledge_base="engineering-handbook",
        question="Which protocol encrypts client data?",
        expected_outcome="no_answer",
        expected_source_filename=None,
        expected_canonical_fact=None,
        outside_matching_evidence=None,
        unavailable_matching_evidence=None,
    )


def source_payload(filename: str = "engineering-handbook.txt") -> dict[str, object]:
    return {"outcome": "results", "results": [{"source_filename": filename}]}


def answer_payload(
    answer: str = "Automated checks must pass before a release.",
    filename: str = "engineering-handbook.txt",
) -> dict[str, object]:
    return {
        "outcome": "answer",
        "answer": answer,
        "references": [
            {
                "reference_key": "source-1",
                "source_filename": filename,
                "file_type": "txt",
                "excerpt": "Sanitized fixture excerpt.",
            }
        ],
    }


def no_answer_payload() -> dict[str, object]:
    return {"outcome": "no_answer", "answer": None, "references": []}


def test_answer_classification_uses_top_five_reference_and_normalized_fact() -> None:
    result = classify_case_result(
        answer_case(),
        READY_KNOWLEDGE_BASE,
        search_status=200,
        search_payload=source_payload(),
        answer_status=200,
        answer_payload=answer_payload(" AUTOMATED\nchecks must pass before a RELEASE. "),
        latency_ms={"source_search": 10, "answer": 20, "total": 30},
    )

    assert result == {
        "case_id": "direct-evidence-001",
        "expected_outcome": "answer",
        "actual_outcome": "answer",
        "expected_source_top_five": "hit",
        "reference_validity": "valid",
        "knowledge_base_isolation": "pass",
        "canonical_fact": "pass",
        "latency_ms": {"source_search": 10, "answer": 20, "total": 30},
        "reason": "passed",
    }


def test_answerable_false_no_answer_is_quality_evidence_not_an_assertion() -> None:
    result = classify_case_result(
        answer_case(),
        READY_KNOWLEDGE_BASE,
        search_status=200,
        search_payload=source_payload(),
        answer_status=200,
        answer_payload=no_answer_payload(),
        latency_ms={"source_search": 1, "answer": 2, "total": 3},
    )

    assert result["actual_outcome"] == "no_answer"
    assert result["reason"] == "false_no_answer"
    assert result["canonical_fact"] == "not_evaluated"


def test_no_answer_that_generates_an_answer_is_classified_as_unsupported() -> None:
    result = classify_case_result(
        no_answer_case(),
        READY_KNOWLEDGE_BASE,
        search_status=200,
        search_payload={"outcome": "no_results", "results": []},
        answer_status=200,
        answer_payload=answer_payload(),
        latency_ms={"source_search": 1, "answer": 2, "total": 3},
    )

    assert result["reason"] == "unsupported_answer"
    assert result["reference_validity"] == "invalid"


@pytest.mark.parametrize(
    "reference",
    [
        {},
        {"reference_key": "source-1", "source_filename": "engineering-handbook.txt"},
        {
            "reference_key": "source-1",
            "source_filename": "engineering-handbook.txt",
            "file_type": "md",
            "excerpt": "Sanitized fixture excerpt.",
        },
        {
            "reference_key": "source-1",
            "source_filename": "engineering-handbook.txt",
            "file_type": "txt",
            "excerpt": "x" * 501,
        },
    ],
)
def test_reference_validity_rejects_missing_or_malformed_safe_fields(
    reference: dict[str, object],
) -> None:
    payload = answer_payload()
    payload["references"] = [reference]

    result = classify_case_result(
        answer_case(),
        READY_KNOWLEDGE_BASE,
        search_status=200,
        search_payload=source_payload(),
        answer_status=200,
        answer_payload=payload,
        latency_ms={"source_search": 1, "answer": 1, "total": 2},
    )

    assert result["reference_validity"] == "invalid"


def test_reference_validity_rejects_duplicate_or_disallowed_sources() -> None:
    duplicate_payload = answer_payload()
    duplicate_payload["references"] = [
        duplicate_payload["references"][0],
        duplicate_payload["references"][0],
    ]
    cross_knowledge_base_payload = answer_payload(filename="isolation-beta.txt")
    non_ready_knowledge_base = FixtureKnowledgeBase(
        "unavailable-evidence",
        (
            FixtureDocument(
                "unavailable-uploaded.txt", "knowledge_bases/unavailable-uploaded.txt", "Uploaded"
            ),
        ),
    )
    non_ready_payload = answer_payload(filename="unavailable-uploaded.txt")

    for knowledge_base, payload in (
        (READY_KNOWLEDGE_BASE, duplicate_payload),
        (READY_KNOWLEDGE_BASE, cross_knowledge_base_payload),
        (non_ready_knowledge_base, non_ready_payload),
    ):
        result = classify_case_result(
            answer_case(),
            knowledge_base,
            search_status=200,
            search_payload=source_payload(),
            answer_status=200,
            answer_payload=payload,
            latency_ms={"source_search": 1, "answer": 1, "total": 2},
        )
        assert result["reference_validity"] == "invalid"


def test_cross_knowledge_base_source_is_an_isolation_failure() -> None:
    case = no_answer_case(category="knowledge_base_isolation")
    case = EvaluationCase(
        **{
            **case.__dict__,
            "outside_matching_evidence": {
                "knowledge_base": "isolation-beta",
                "source_filename": "isolation-beta.txt",
                "canonical_fact": "Harbor deployment policy",
            },
        }
    )
    result = classify_case_result(
        case,
        READY_KNOWLEDGE_BASE,
        search_status=200,
        search_payload=source_payload("isolation-beta.txt"),
        answer_status=200,
        answer_payload=no_answer_payload(),
        latency_ms={"source_search": 1, "answer": 2, "total": 3},
    )

    assert result["knowledge_base_isolation"] == "fail"
    assert result["reason"] == "knowledge_base_isolation_failure"


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def post_json(self, path: str, _payload: dict[str, str]):
        self.calls.append(path)
        if "sources/search" in path:
            return 200, source_payload()
        if len(self.calls) == 2:
            return None, None
        return 200, answer_payload()


class ThrowingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.raised = False

    def post_json(self, path: str, payload: dict[str, str]):
        if not self.raised and len(self.calls) == 1:
            self.raised = True
            raise RuntimeError("private infrastructure detail")
        return super().post_json(path, payload)


def test_one_failed_case_does_not_stop_subsequent_cases() -> None:
    first = answer_case()
    second = EvaluationCase(
        **{**first.__dict__, "case_id": "direct-evidence-002", "question": "Who owns it?"}
    )
    manifest = type(
        "Manifest",
        (),
        {
            "cases": (first, second),
            "knowledge_bases": {"engineering-handbook": READY_KNOWLEDGE_BASE},
        },
    )()
    client = FakeClient()

    results = run_evaluation(
        manifest,
        client=client,
        knowledge_base_ids={"engineering-handbook": "fixture-id"},
    )

    assert len(results) == 2
    assert results[0]["reason"] == "answer_unavailable"
    assert results[1]["reason"] == "passed"
    assert len(client.calls) == 4


def test_unexpected_case_failure_is_safe_and_does_not_stop_later_cases() -> None:
    first = answer_case()
    second = EvaluationCase(
        **{**first.__dict__, "case_id": "direct-evidence-002", "question": "Who owns it?"}
    )
    manifest = type(
        "Manifest",
        (),
        {
            "cases": (first, second),
            "knowledge_bases": {"engineering-handbook": READY_KNOWLEDGE_BASE},
        },
    )()
    client = ThrowingClient()

    results = run_evaluation(
        manifest,
        client=client,
        knowledge_base_ids={"engineering-handbook": "fixture-id"},
    )

    assert [result["reason"] for result in results] == [
        "answer_unavailable",
        "passed",
    ]


def test_report_contains_only_safe_result_fields_and_summary_metrics(tmp_path: Path) -> None:
    result = classify_case_result(
        answer_case(),
        READY_KNOWLEDGE_BASE,
        search_status=200,
        search_payload=source_payload(),
        answer_status=200,
        answer_payload=answer_payload(),
        latency_ms={"source_search": 1, "answer": 2, "total": 3},
    )
    manifest = type("Manifest", (), {"schema_version": 1})()
    report = build_report(manifest, [result], cold_answer_ms=11, cached_answer_ms=7)
    report_path, summary_path = write_reports(tmp_path, report)
    serialized = report_path.read_text(encoding="utf-8")

    assert json.loads(serialized)["summary"]["valid_reference_rate"] == 1.0
    assert summary_path.is_file()
    for forbidden in (
        "What must happen",
        "Automated checks",
        "http://",
        "postgresql",
        "qdrant",
        "password",
    ):
        assert forbidden not in serialized


def test_loading_the_manifest_does_not_modify_the_committed_fixture(tmp_path: Path) -> None:
    manifest_path = Path(__file__).parent / "fixtures" / "grounded_answers" / "v1" / "manifest.json"
    before = manifest_path.read_bytes()

    manifest = load_manifest(manifest_path)

    assert manifest.schema_version == 1
    assert len(manifest.cases) == 20
    assert manifest_path.read_bytes() == before


class CacheClient:
    def ollama_models(self):
        return [
            {
                "name": "qwen2.5:3b-instruct-q4_K_M",
                "digest": "sha256:357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b",
            }
        ]


def test_preflight_requires_the_pinned_bge_revision_and_ollama_digest(tmp_path: Path) -> None:
    snapshot = (
        tmp_path
        / "hub"
        / "models--BAAI--bge-m3"
        / "snapshots"
        / "5617a9f61b028005a4858fdac845db406aefb181"
    )
    snapshot.mkdir(parents=True)

    preflight_model_caches(CacheClient(), tmp_path)


def test_preflight_rejects_a_missing_pinned_bge_revision(tmp_path: Path) -> None:
    with pytest.raises(evaluation.EvaluationError, match="Run the documented prewarm command"):
        preflight_model_caches(CacheClient(), tmp_path)


@pytest.mark.parametrize(
    "models",
    [
        [],
        [{"name": "qwen2.5:3b-instruct-q4_K_M", "digest": "sha256:incorrect"}],
    ],
)
def test_preflight_rejects_a_missing_or_changed_ollama_model(
    tmp_path: Path, models: list[dict[str, str]]
) -> None:
    snapshot = (
        tmp_path
        / "hub"
        / "models--BAAI--bge-m3"
        / "snapshots"
        / "5617a9f61b028005a4858fdac845db406aefb181"
    )
    snapshot.mkdir(parents=True)

    class MissingModelClient:
        def ollama_models(self):
            return models

    with pytest.raises(evaluation.EvaluationError, match="Run the documented prewarm command"):
        preflight_model_caches(MissingModelClient(), tmp_path)


def test_preflight_failure_stops_before_seeding_or_http_evaluation(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version": 1, "knowledge_bases": [], "cases": []}', encoding="utf-8"
    )
    calls: list[str] = []

    class NoTrafficClient:
        def __init__(self, _api_url: str) -> None:
            calls.append("client")

    def fail_preflight(_client, _cache) -> None:
        raise evaluation.EvaluationError(
            "The required BGE-M3 evaluation cache is unavailable. "
            "Run the documented prewarm command."
        )

    monkeypatch.setattr(evaluation, "RequestsEvaluationClient", NoTrafficClient)
    monkeypatch.setattr(evaluation, "preflight_model_caches", fail_preflight)
    monkeypatch.setattr(
        evaluation,
        "seed_fixture_stack",
        lambda *_args, **_kwargs: pytest.fail("fixture seeding must not run"),
    )
    monkeypatch.setattr(
        evaluation,
        "run_evaluation",
        lambda *_args, **_kwargs: pytest.fail("HTTP evaluation must not run"),
    )

    with pytest.raises(evaluation.EvaluationError, match="Run the documented prewarm command"):
        evaluation.run_from_arguments(
            Namespace(
                manifest=manifest_path,
                output_directory=tmp_path / "output",
                api_url="http://isolated-api",
                embedding_cache=tmp_path / "cache",
            )
        )

    assert calls == ["client"]


def test_non_ready_seeder_creates_document_and_job_without_dispatch(
    monkeypatch, tmp_path: Path
) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "grounded_answers" / "v1"
    manifest = load_manifest(fixture_root / "manifest.json")
    knowledge_base_id = uuid4()
    session = evaluation.SessionLocal()
    session.add(
        KnowledgeBase(
            id=knowledge_base_id,
            name="Evaluation unavailable evidence",
            normalized_name="evaluation unavailable evidence",
        )
    )
    session.commit()
    session.close()

    class Storage:
        def upload_file(self, _file, _object_key, _content_type) -> None:
            return None

    monkeypatch.setattr(evaluation, "get_storage_client", lambda: Storage())
    evaluation._seed_non_ready_documents(
        manifest,
        fixture_root,
        {label: str(knowledge_base_id) for label in manifest.knowledge_bases},
    )

    session = evaluation.SessionLocal()
    try:
        documents = session.scalars(
            select(Document).where(Document.status.in_(["Uploaded", "Failed"]))
        ).all()
        jobs = session.scalars(select(DocumentProcessingJob)).all()
    finally:
        session.close()

    assert {document.status for document in documents} == {"Uploaded", "Failed"}
    assert {job.state for job in jobs} == {"Queued", "Completed"}
    assert all(job.celery_task_id is None for job in jobs)
