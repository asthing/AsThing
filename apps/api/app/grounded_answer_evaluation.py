"""Run the committed grounded-answer fixtures against an isolated local stack.

This module is an evaluation tool. It is not imported by the API routes or worker.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import requests
from sqlalchemy import select

from app.database import SessionLocal
from app.document_models import (
    Document,
    DocumentProcessingJob,
    ProcessingJobState,
)
from app.embeddings import EMBEDDING_MODEL_REVISION
from app.knowledge_bases import KnowledgeBase
from app.local_llm import OLLAMA_MODEL, OLLAMA_MODEL_DIGEST
from app.storage import ObjectStorageError, get_storage_client

REPORT_SCHEMA_VERSION = 1
ANSWERABLE_CATEGORIES = {"direct_evidence", "paraphrased_evidence"}
ALLOWED_REFERENCE_FILE_TYPES = {"pdf", "txt", "docx"}
MAX_REFERENCE_EXCERPT_LENGTH = 500
SAFE_REASONS = {
    "passed",
    "false_no_answer",
    "unsupported_answer",
    "expected_source_missing",
    "invalid_references",
    "knowledge_base_isolation_failure",
    "canonical_fact_missing",
    "source_search_unavailable",
    "answer_unavailable",
    "invalid_response",
}
DEFAULT_REQUEST_TIMEOUT_SECONDS = 190
READY_WAIT_TIMEOUT_SECONDS = 900
READY_WAIT_INTERVAL_SECONDS = 2


class EvaluationError(Exception):
    """Raised for a safe setup or preflight failure."""


class EvaluationHttpClient(Protocol):
    def post_json(
        self, path: str, payload: dict[str, str]
    ) -> tuple[int | None, dict[str, Any] | None]: ...


@dataclass(frozen=True)
class FixtureDocument:
    filename: str
    path: str
    status: str


@dataclass(frozen=True)
class FixtureKnowledgeBase:
    label: str
    documents: tuple[FixtureDocument, ...]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    split: str
    category: str
    knowledge_base: str
    question: str
    expected_outcome: str
    expected_source_filename: str | None
    expected_canonical_fact: str | None
    outside_matching_evidence: dict[str, str] | None
    unavailable_matching_evidence: dict[str, str] | None


@dataclass(frozen=True)
class EvaluationManifest:
    schema_version: int
    knowledge_bases: dict[str, FixtureKnowledgeBase]
    cases: tuple[EvaluationCase, ...]


class RequestsEvaluationClient:
    def __init__(
        self, base_url: str, *, timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def post_json(
        self, path: str, payload: dict[str, str]
    ) -> tuple[int | None, dict[str, Any] | None]:
        try:
            response = self.session.post(
                f"{self.base_url}{path}", json=payload, timeout=self.timeout_seconds
            )
            data = response.json()
            return response.status_code, data if isinstance(data, dict) else None
        except (requests.RequestException, ValueError):
            return None, None

    def create_knowledge_base(self, name: str) -> str:
        status_code, payload = self.post_json("/knowledge-bases", {"name": name})
        if (
            status_code != 201
            or not isinstance(payload, dict)
            or not isinstance(payload.get("id"), str)
        ):
            raise EvaluationError("Fixture knowledge bases could not be created.")
        return payload["id"]

    def upload_fixture(
        self, knowledge_base_id: str, document: FixtureDocument, content: bytes
    ) -> None:
        try:
            response = self.session.post(
                f"{self.base_url}/knowledge-bases/{knowledge_base_id}/documents",
                files={"file": (document.filename, content, "text/plain")},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException:
            raise EvaluationError("Ready fixture documents could not be uploaded.") from None
        if response.status_code != 201:
            raise EvaluationError("Ready fixture documents could not be uploaded.")

    def list_documents(self, knowledge_base_id: str) -> list[dict[str, Any]]:
        try:
            response = self.session.get(
                f"{self.base_url}/knowledge-bases/{knowledge_base_id}/documents",
                timeout=20,
            )
            payload = response.json()
        except (requests.RequestException, ValueError):
            raise EvaluationError("Ready fixture processing could not be verified.") from None
        if response.status_code != 200 or not isinstance(payload, list):
            raise EvaluationError("Ready fixture processing could not be verified.")
        return [item for item in payload if isinstance(item, dict)]

    def ollama_models(self) -> list[dict[str, Any]]:
        try:
            response = self.session.get("http://ollama:11434/api/tags", timeout=10)
            payload = response.json()
        except (requests.RequestException, ValueError):
            raise EvaluationError("The required local Ollama model cache is unavailable.") from None
        models = payload.get("models") if isinstance(payload, dict) else None
        if response.status_code != 200 or not isinstance(models, list):
            raise EvaluationError("The required local Ollama model cache is unavailable.")
        return [model for model in models if isinstance(model, dict)]


def load_manifest(path: Path) -> EvaluationManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise EvaluationError("The committed evaluation manifest could not be loaded.") from None

    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise EvaluationError("The evaluation manifest version is not supported.")
    raw_knowledge_bases = raw.get("knowledge_bases")
    raw_cases = raw.get("cases")
    if not isinstance(raw_knowledge_bases, list) or not isinstance(raw_cases, list):
        raise EvaluationError("The evaluation manifest is invalid.")

    knowledge_bases: dict[str, FixtureKnowledgeBase] = {}
    for raw_knowledge_base in raw_knowledge_bases:
        if not isinstance(raw_knowledge_base, dict):
            raise EvaluationError("The evaluation manifest is invalid.")
        label = raw_knowledge_base.get("id")
        raw_documents = raw_knowledge_base.get("documents")
        if (
            not isinstance(label, str)
            or not isinstance(raw_documents, list)
            or label in knowledge_bases
        ):
            raise EvaluationError("The evaluation manifest is invalid.")
        documents: list[FixtureDocument] = []
        for raw_document in raw_documents:
            if not isinstance(raw_document, dict):
                raise EvaluationError("The evaluation manifest is invalid.")
            filename = raw_document.get("filename")
            fixture_path = raw_document.get("path")
            status = raw_document.get("status")
            if not all(
                isinstance(value, str) and value for value in (filename, fixture_path, status)
            ):
                raise EvaluationError("The evaluation manifest is invalid.")
            documents.append(FixtureDocument(filename, fixture_path, status))
        knowledge_bases[label] = FixtureKnowledgeBase(label, tuple(documents))

    cases: list[EvaluationCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise EvaluationError("The evaluation manifest is invalid.")
        required = ("id", "split", "category", "knowledge_base", "question", "expected_outcome")
        if any(not isinstance(raw_case.get(key), str) or not raw_case[key] for key in required):
            raise EvaluationError("The evaluation manifest is invalid.")
        if raw_case["knowledge_base"] not in knowledge_bases:
            raise EvaluationError("The evaluation manifest is invalid.")
        cases.append(
            EvaluationCase(
                case_id=raw_case["id"],
                split=raw_case["split"],
                category=raw_case["category"],
                knowledge_base=raw_case["knowledge_base"],
                question=raw_case["question"],
                expected_outcome=raw_case["expected_outcome"],
                expected_source_filename=_optional_string(raw_case, "expected_source_filename"),
                expected_canonical_fact=_optional_string(raw_case, "expected_canonical_fact"),
                outside_matching_evidence=_optional_metadata(raw_case, "outside_matching_evidence"),
                unavailable_matching_evidence=_optional_metadata(
                    raw_case, "unavailable_matching_evidence"
                ),
            )
        )
    return EvaluationManifest(1, knowledge_bases, tuple(cases))


def run_evaluation(
    manifest: EvaluationManifest,
    *,
    client: EvaluationHttpClient,
    knowledge_base_ids: dict[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in manifest.cases:
        started = time.monotonic()
        search_status: int | None = None
        search_payload: dict[str, Any] | None = None
        answer_status: int | None = None
        answer_payload: dict[str, Any] | None = None
        search_latency_ms = 0
        answer_latency_ms = 0
        try:
            search_started = time.monotonic()
            search_status, search_payload = client.post_json(
                f"/knowledge-bases/{knowledge_base_ids[case.knowledge_base]}/sources/search",
                {"question": case.question},
            )
            search_latency_ms = _elapsed_ms(search_started)
            answer_started = time.monotonic()
            answer_status, answer_payload = client.post_json(
                f"/knowledge-bases/{knowledge_base_ids[case.knowledge_base]}/answers",
                {"question": case.question},
            )
            answer_latency_ms = _elapsed_ms(answer_started)
        except Exception:
            # The result below intentionally stores only a fixed safe category.
            pass
        results.append(
            classify_case_result(
                case,
                manifest.knowledge_bases[case.knowledge_base],
                search_status=search_status,
                search_payload=search_payload,
                answer_status=answer_status,
                answer_payload=answer_payload,
                latency_ms={
                    "source_search": search_latency_ms,
                    "answer": answer_latency_ms,
                    "total": _elapsed_ms(started),
                },
            )
        )
    return results


def classify_case_result(
    case: EvaluationCase,
    knowledge_base: FixtureKnowledgeBase,
    *,
    search_status: int | None,
    search_payload: dict[str, Any] | None,
    answer_status: int | None,
    answer_payload: dict[str, Any] | None,
    latency_ms: dict[str, int],
) -> dict[str, Any]:
    ready_sources = {
        document.filename for document in knowledge_base.documents if document.status == "Ready"
    }
    result = {
        "case_id": case.case_id,
        "expected_outcome": case.expected_outcome,
        "actual_outcome": _actual_outcome(answer_status, answer_payload),
        "expected_source_top_five": _expected_source_result(case, search_status, search_payload),
        "reference_validity": _reference_validity(
            case, ready_sources, answer_status, answer_payload
        ),
        "knowledge_base_isolation": _isolation_result(
            case, ready_sources, search_status, search_payload, answer_status, answer_payload
        ),
        "canonical_fact": _canonical_fact_result(case, answer_status, answer_payload),
        "latency_ms": {key: max(0, int(value)) for key, value in latency_ms.items()},
    }
    result["reason"] = _safe_reason(result)
    return result


def build_report(
    manifest: EvaluationManifest,
    results: list[dict[str, Any]],
    *,
    cold_answer_ms: int | None,
    cached_answer_ms: int | None,
) -> dict[str, Any]:
    answerable = [result for result in results if result["expected_outcome"] == "answer"]
    no_answer_cases = [result for result in results if result["expected_outcome"] == "no_answer"]
    answer_responses = [result for result in results if result["actual_outcome"] == "answer"]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest_schema_version": manifest.schema_version,
        "case_count": len(results),
        "results": results,
        "summary": {
            "retrieval_top_five_hit_rate": _rate(
                sum(result["expected_source_top_five"] == "hit" for result in answerable),
                len(answerable),
            ),
            "answerable_case_completion_rate": _rate(
                sum(_is_completed_answer(result) for result in answerable), len(answerable)
            ),
            "false_no_answer_count": sum(
                result["actual_outcome"] == "no_answer" for result in answerable
            ),
            "unsupported_answer_count": sum(
                result["actual_outcome"] == "answer" for result in no_answer_cases
            ),
            "valid_reference_rate": _rate(
                sum(result["reference_validity"] == "valid" for result in answer_responses),
                len(answer_responses),
            ),
            "knowledge_base_isolation_failures": sum(
                result["knowledge_base_isolation"] == "fail" for result in results
            ),
            "cold_answer_ms": cold_answer_ms,
            "cached_answer_ms": cached_answer_ms,
        },
    }
    return report


def write_reports(output_directory: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "report.json"
    summary_path = output_directory / "summary.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = report["summary"]
    summary_path.write_text(
        "\n".join(
            [
                "# Grounded-answer evaluation summary",
                "",
                f"Cases: {report['case_count']}",
                f"Top-five retrieval hit rate: {summary['retrieval_top_five_hit_rate']}",
                f"Answerable-case completion rate: {summary['answerable_case_completion_rate']}",
                f"False no-answer count: {summary['false_no_answer_count']}",
                f"Unsupported-answer count: {summary['unsupported_answer_count']}",
                f"Valid-reference rate: {summary['valid_reference_rate']}",
                (
                    "Knowledge-base-isolation failures: "
                    f"{summary['knowledge_base_isolation_failures']}"
                ),
                f"Cold answer observation (ms): {summary['cold_answer_ms']}",
                f"Cached answer observation (ms): {summary['cached_answer_ms']}",
                "",
                (
                    "Cold means the first generation after service startup with the model "
                    "already cached."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path, summary_path


def preflight_model_caches(client: RequestsEvaluationClient, embedding_cache: Path) -> None:
    bge_snapshot = (
        embedding_cache / "hub" / "models--BAAI--bge-m3" / "snapshots" / EMBEDDING_MODEL_REVISION
    )
    if not bge_snapshot.is_dir():
        raise EvaluationError(
            "The required BGE-M3 evaluation cache is unavailable. "
            "Run the documented prewarm command."
        )
    required_digest = OLLAMA_MODEL_DIGEST.removeprefix("sha256:")
    if not any(
        model.get("name") == OLLAMA_MODEL
        and str(model.get("digest", "")).removeprefix("sha256:") == required_digest
        for model in client.ollama_models()
    ):
        raise EvaluationError(
            "The required Ollama evaluation cache is unavailable. "
            "Run the documented prewarm command."
        )


def seed_fixture_stack(
    manifest: EvaluationManifest,
    *,
    fixture_root: Path,
    client: RequestsEvaluationClient,
) -> dict[str, str]:
    knowledge_base_ids = {
        label: client.create_knowledge_base(f"Evaluation {label}")
        for label in manifest.knowledge_bases
    }
    for label, knowledge_base in manifest.knowledge_bases.items():
        for document in knowledge_base.documents:
            if document.status == "Ready":
                client.upload_fixture(
                    knowledge_base_ids[label], document, _fixture_bytes(fixture_root, document.path)
                )
    _wait_for_ready_documents(manifest, knowledge_base_ids, client)
    _seed_non_ready_documents(manifest, fixture_root, knowledge_base_ids)
    return knowledge_base_ids


def run_from_arguments(arguments: argparse.Namespace) -> int:
    manifest = load_manifest(arguments.manifest)
    client = RequestsEvaluationClient(arguments.api_url)
    preflight_model_caches(client, arguments.embedding_cache)
    knowledge_base_ids = seed_fixture_stack(
        manifest, fixture_root=arguments.manifest.parent, client=client
    )
    timing_case = next(case for case in manifest.cases if case.category in ANSWERABLE_CATEGORIES)
    cold_answer_ms = _answer_latency(
        client, knowledge_base_ids[timing_case.knowledge_base], timing_case.question
    )
    cached_answer_ms = _answer_latency(
        client, knowledge_base_ids[timing_case.knowledge_base], timing_case.question
    )
    results = run_evaluation(manifest, client=client, knowledge_base_ids=knowledge_base_ids)
    report = build_report(
        manifest, results, cold_answer_ms=cold_answer_ms, cached_answer_ms=cached_answer_ms
    )
    report_path, summary_path = write_reports(arguments.output_directory, report)
    print(f"Evaluation completed: {len(results)} cases")
    print(f"Report: {report_path.name}")
    print(f"Summary: {summary_path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local grounded-answer evaluation.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--api-url", default="http://api:8000")
    parser.add_argument("--embedding-cache", type=Path, default=Path("/evaluation-embedding-cache"))
    parser.add_argument("--prewarm-embedding-cache", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.prewarm_embedding_cache:
            _prewarm_embedding_cache()
            print("BGE-M3 evaluation cache prewarm complete.")
            return 0
        if arguments.manifest is None or arguments.output_directory is None:
            raise EvaluationError("A manifest and output directory are required for evaluation.")
        return run_from_arguments(arguments)
    except EvaluationError as error:
        print(error, file=sys.stderr)
        return 2


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is not None and not isinstance(value, str):
        raise EvaluationError("The evaluation manifest is invalid.")
    return value


def _optional_metadata(raw: dict[str, Any], key: str) -> dict[str, str] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value.values()):
        raise EvaluationError("The evaluation manifest is invalid.")
    return dict(value)


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _actual_outcome(status_code: int | None, payload: dict[str, Any] | None) -> str:
    if status_code == 200 and payload and payload.get("outcome") in {"answer", "no_answer"}:
        return str(payload["outcome"])
    return "error"


def _expected_source_result(
    case: EvaluationCase, status_code: int | None, payload: dict[str, Any] | None
) -> str:
    if case.expected_source_filename is None:
        return "not_applicable"
    if status_code != 200 or not payload or not isinstance(payload.get("results"), list):
        return "not_applicable"
    filenames = {
        item.get("source_filename")
        for item in payload["results"]
        if isinstance(item, dict) and isinstance(item.get("source_filename"), str)
    }
    return "hit" if case.expected_source_filename in filenames else "miss"


def _reference_validity(
    case: EvaluationCase,
    ready_sources: set[str],
    status_code: int | None,
    payload: dict[str, Any] | None,
) -> str:
    if status_code != 200 or not payload:
        return "not_applicable"
    references = payload.get("references")
    if not isinstance(references, list):
        return "invalid"
    if payload.get("outcome") == "no_answer":
        return "not_applicable" if not references else "invalid"
    if (
        payload.get("outcome") != "answer"
        or not references
        or case.expected_source_filename is None
    ):
        return "invalid"
    reference_keys: set[str] = set()
    for reference in references:
        if not isinstance(reference, dict):
            return "invalid"
        reference_key = reference.get("reference_key")
        source_filename = reference.get("source_filename")
        file_type = reference.get("file_type")
        excerpt = reference.get("excerpt")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (reference_key, source_filename, file_type, excerpt)
        ):
            return "invalid"
        if (
            reference_key in reference_keys
            or file_type not in ALLOWED_REFERENCE_FILE_TYPES
            or len(excerpt) > MAX_REFERENCE_EXCERPT_LENGTH
            or source_filename not in ready_sources
        ):
            return "invalid"
        reference_keys.add(reference_key)
    return "valid"


def _isolation_result(
    case: EvaluationCase,
    ready_sources: set[str],
    search_status: int | None,
    search_payload: dict[str, Any] | None,
    answer_status: int | None,
    answer_payload: dict[str, Any] | None,
) -> str:
    if search_status != 200 or answer_status != 200 or not search_payload or not answer_payload:
        return "not_evaluated"
    result_filenames = {
        item.get("source_filename")
        for item in search_payload.get("results", [])
        if isinstance(item, dict) and isinstance(item.get("source_filename"), str)
    }
    reference_filenames = {
        item.get("source_filename")
        for item in answer_payload.get("references", [])
        if isinstance(item, dict) and isinstance(item.get("source_filename"), str)
    }
    if not result_filenames <= ready_sources or not reference_filenames <= ready_sources:
        return "fail"
    if case.outside_matching_evidence:
        outside_source = case.outside_matching_evidence["source_filename"]
        if outside_source in result_filenames or outside_source in reference_filenames:
            return "fail"
        if answer_payload.get("outcome") != "no_answer":
            return "fail"
    if case.unavailable_matching_evidence and answer_payload.get("outcome") != "no_answer":
        return "fail"
    return "pass"


def _canonical_fact_result(
    case: EvaluationCase, status_code: int | None, payload: dict[str, Any] | None
) -> str:
    if case.expected_canonical_fact is None:
        return "not_applicable"
    if status_code != 200 or not payload or payload.get("outcome") != "answer":
        return "not_evaluated"
    answer = payload.get("answer")
    if not isinstance(answer, str):
        return "fail"
    return "pass" if _normalized(case.expected_canonical_fact) in _normalized(answer) else "fail"


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _safe_reason(result: dict[str, Any]) -> str:
    if result["actual_outcome"] == "error":
        if result["expected_source_top_five"] == "not_applicable":
            return "source_search_unavailable"
        return "answer_unavailable"
    if (
        result["expected_source_top_five"] == "not_applicable"
        and result["expected_outcome"] == "answer"
    ):
        return "source_search_unavailable"
    if result["knowledge_base_isolation"] == "fail":
        return "knowledge_base_isolation_failure"
    if result["expected_outcome"] == "answer" and result["actual_outcome"] == "no_answer":
        return "false_no_answer"
    if result["expected_outcome"] == "no_answer" and result["actual_outcome"] == "answer":
        return "unsupported_answer"
    if result["expected_source_top_five"] == "miss":
        return "expected_source_missing"
    if result["reference_validity"] == "invalid":
        return "invalid_references"
    if result["canonical_fact"] == "fail":
        return "canonical_fact_missing"
    return "passed"


def _is_completed_answer(result: dict[str, Any]) -> bool:
    return (
        result["actual_outcome"] == "answer"
        and result["expected_source_top_five"] == "hit"
        and result["reference_validity"] == "valid"
        and result["knowledge_base_isolation"] == "pass"
        and result["canonical_fact"] == "pass"
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _fixture_bytes(fixture_root: Path, relative_path: str) -> bytes:
    candidate = (fixture_root / relative_path).resolve()
    if not candidate.is_relative_to(fixture_root.resolve()) or not candidate.is_file():
        raise EvaluationError("An evaluation fixture file is unavailable.")
    return candidate.read_bytes()


def _wait_for_ready_documents(
    manifest: EvaluationManifest,
    knowledge_base_ids: dict[str, str],
    client: RequestsEvaluationClient,
) -> None:
    expected = {
        label: {
            document.filename for document in knowledge_base.documents if document.status == "Ready"
        }
        for label, knowledge_base in manifest.knowledge_bases.items()
    }
    deadline = time.monotonic() + READY_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        all_ready = True
        for label, filenames in expected.items():
            documents = client.list_documents(knowledge_base_ids[label])
            statuses = {
                item.get("original_filename"): item.get("status")
                for item in documents
                if isinstance(item.get("original_filename"), str)
            }
            if any(statuses.get(filename) != "Ready" for filename in filenames):
                all_ready = False
                break
        if all_ready:
            return
        time.sleep(READY_WAIT_INTERVAL_SECONDS)
    raise EvaluationError(
        "Ready fixture processing did not complete before the evaluation timeout."
    )


def _seed_non_ready_documents(
    manifest: EvaluationManifest, fixture_root: Path, knowledge_base_ids: dict[str, str]
) -> None:
    # The worker can finish its last Ready-document transaction immediately before this
    # isolated test-only seeder starts. Retry the whole atomic seed once so that transient
    # local service/database startup races do not invalidate an evaluation run.
    for attempt in range(2):
        try:
            _seed_non_ready_documents_once(manifest, fixture_root, knowledge_base_ids)
            return
        except EvaluationError:
            if attempt == 1:
                raise
            time.sleep(1)


def _seed_non_ready_documents_once(
    manifest: EvaluationManifest, fixture_root: Path, knowledge_base_ids: dict[str, str]
) -> None:
    now = datetime.now(UTC)
    session = SessionLocal()
    try:
        valid_knowledge_base_ids = set(session.scalars(select(KnowledgeBase.id)).all())
        if any(
            UUID(knowledge_base_id) not in valid_knowledge_base_ids
            for knowledge_base_id in knowledge_base_ids.values()
        ):
            raise EvaluationError("Evaluation fixture knowledge bases could not be verified.")
        for label, knowledge_base in manifest.knowledge_bases.items():
            for document in knowledge_base.documents:
                if document.status == "Ready":
                    continue
                content = _fixture_bytes(fixture_root, document.path)
                document_id = uuid4()
                object_key = f"evaluation-fixtures/{document_id}/{document.filename}"
                try:
                    get_storage_client().upload_file(BytesIO(content), object_key, "text/plain")
                except ObjectStorageError:
                    raise EvaluationError(
                        "Non-Ready fixture documents could not be seeded."
                    ) from None
                seeded_document = Document(
                    id=document_id,
                    knowledge_base_id=UUID(knowledge_base_ids[label]),
                    original_filename=document.filename,
                    file_size_bytes=len(content),
                    file_type="txt",
                    status=document.status,
                    storage_object_key=object_key,
                    processing_started_at=now if document.status == "Failed" else None,
                    processing_completed_at=now if document.status == "Failed" else None,
                    failure_code="EVALUATION_SEEDED_FAILURE"
                    if document.status == "Failed"
                    else None,
                    failure_message=(
                        "Evaluation fixture intentionally unavailable."
                        if document.status == "Failed"
                        else None
                    ),
                    processing_attempt_count=1 if document.status == "Failed" else 0,
                )
                seeded_job = DocumentProcessingJob(
                    id=uuid4(),
                    document_id=document_id,
                    state=(
                        ProcessingJobState.COMPLETED
                        if document.status == "Failed"
                        else ProcessingJobState.QUEUED
                    ),
                    attempt_count=1 if document.status == "Failed" else 0,
                    failure_code="EVALUATION_SEEDED_FAILURE"
                    if document.status == "Failed"
                    else None,
                    failure_message=(
                        "Evaluation fixture intentionally unavailable."
                        if document.status == "Failed"
                        else None
                    ),
                    started_at=now if document.status == "Failed" else None,
                    completed_at=now if document.status == "Failed" else None,
                )
                # These models intentionally have no ORM relationship. Flush the parent first so
                # PostgreSQL always sees the document before its required processing-job record.
                session.add(seeded_document)
                session.flush()
                session.add(seeded_job)
        session.commit()
    except EvaluationError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        raise EvaluationError("Non-Ready fixture database records could not be seeded.") from None
    finally:
        session.close()


def _answer_latency(
    client: EvaluationHttpClient, knowledge_base_id: str, question: str
) -> int | None:
    started = time.monotonic()
    status_code, _ = client.post_json(
        f"/knowledge-bases/{knowledge_base_id}/answers", {"question": question}
    )
    return _elapsed_ms(started) if status_code == 200 else None


def _prewarm_embedding_cache() -> None:
    # This mode is invoked only by the documented manual prewarm command.
    from app.embeddings import LocalEmbeddingService

    LocalEmbeddingService().embed(["evaluation cache prewarm"])


if __name__ == "__main__":
    raise SystemExit(main())
