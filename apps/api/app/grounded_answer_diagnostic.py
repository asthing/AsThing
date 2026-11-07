"""Opt-in fixture diagnostics; never imported by production routes or workers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import requests
from pydantic import ValidationError

from app.answers import AnswerService
from app.grounded_answer_evaluation import (
    EvaluationCase,
    EvaluationError,
    EvaluationManifest,
    RequestsEvaluationClient,
    _canonical_fact_result,
    _normalized,
    _reference_validity,
    load_manifest,
    preflight_model_caches,
    seed_fixture_stack,
)
from app.local_llm import (
    LocalAnswerModel,
    LocalModelError,
    LocalModelErrorCategory,
    ModelDecision,
)
from app.retrieval import RetrievalDependencyError, SourceRetrievalService

CLASSIFICATIONS = (
    "matched", "retrieval_miss", "answer_context_missing_fact", "false_abstention",
    "invalid_reference", "invalid_model_output", "semantic_match_uncertain",
    "canonical_fact_omission",
)
EXECUTION_STATUSES = {"ok", "dependency_unavailable", "unexpected_failure"}
VALIDATION_CATEGORIES = {"none", "reference", "structured_output"}
REVIEW_DECISIONS = {
    "retrieved_evidence_missing", "context_evidence_missing", "context_support_confirmed",
    "answer_fact_omitted",
}
PREFLIGHT_MESSAGE = "Diagnostic preflight failed. Run the documented evaluation prewarm commands."
CONSOLE_MESSAGE = (
    "Raw console inspection is disabled: this container path cannot guarantee unrecorded output."
)


@dataclass
class Observation:
    expected_source_retrieved: bool = False
    retrieved_fact_present: bool = False
    context_fact_present: bool = False
    retrieval_empty: bool = False
    context_empty: bool = False
    generation_invoked: bool = False
    generation_outcome: str = "not_invoked"
    validation_failure: str = "none"
    execution_status: str = "ok"
    canonical_match: bool = False
    references_valid: bool = False


def contains_fact(case: EvaluationCase, content: str) -> bool:
    return bool(case.expected_canonical_fact) and (
        _normalized(case.expected_canonical_fact) in _normalized(content)
    )


def output_failure(content: object, allowed_keys: set[str]) -> str:
    """Inspect validation without another generation or retaining raw output."""
    if not isinstance(content, str):
        return "structured_output"
    try:
        payload = json.loads(content)
    except ValueError:
        return "structured_output"
    if not isinstance(payload, dict) or payload.get("outcome") not in {"answer", "no_answer"}:
        return "structured_output"
    references = payload.get("references")
    if (
        not isinstance(references, list)
        or any(not isinstance(key, str) or key not in allowed_keys for key in references)
        or len(references) != len(set(references))
        or len(references) > 5
        or (payload["outcome"] == "answer" and not references)
        or (payload["outcome"] == "no_answer" and references)
    ):
        return "reference"
    try:
        ModelDecision.model_validate_json(content)
    except ValidationError:
        return "structured_output"
    return "none"


class ObservedRetriever:
    def __init__(self, delegate, case: EvaluationCase, observation: Observation):
        self.delegate = delegate
        self.case = case
        self.observation = observation
        self.expected_contents: set[str] = set()

    def search_evidence(self, knowledge_base_id, question):
        evidence = self.delegate.search_evidence(knowledge_base_id, question)
        self.expected_contents = {
            chunk.content for chunk in evidence.chunks
            if chunk.source_filename == self.case.expected_source_filename
        }
        self.observation.expected_source_retrieved = bool(self.expected_contents)
        self.observation.retrieved_fact_present = any(
            contains_fact(self.case, content) for content in self.expected_contents
        )
        self.observation.retrieval_empty = not evidence.chunks
        return evidence


class ObservedHttp:
    def __init__(self, delegate, observation: Observation):
        self.delegate = delegate
        self.observation = observation
        self.allowed_keys: set[str] = set()

    def get(self, *args, **kwargs):
        return self.delegate.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        self.observation.generation_invoked = True
        response = self.delegate.post(*args, **kwargs)
        response.raise_for_status()
        try:
            content = response.json()["message"]["content"]
            self.observation.validation_failure = output_failure(content, self.allowed_keys)
        except (ValueError, KeyError, TypeError):
            self.observation.validation_failure = "structured_output"
        return response


class ObservedModel:
    def __init__(self, model, http: ObservedHttp, retriever: ObservedRetriever):
        self.model = model
        self.http = http
        self.retriever = retriever

    def generate(self, question, sources):
        observation = self.retriever.observation
        observation.context_empty = not sources
        observation.context_fact_present = any(
            source.content in self.retriever.expected_contents
            and contains_fact(self.retriever.case, source.content)
            for source in sources
        )
        self.http.allowed_keys = {source.key for source in sources}
        decision = self.model.generate(question, sources)
        observation.generation_outcome = decision.outcome
        return decision


def classify(observation: Observation, review: str | None = None) -> str | None:
    """Precedence: execution, validation, match, confirmed evidence gaps, abstention, review."""
    if observation.execution_status != "ok":
        return None
    if observation.validation_failure == "reference":
        return "invalid_reference"
    if observation.validation_failure == "structured_output":
        return "invalid_model_output"
    if observation.canonical_match and observation.references_valid:
        return "matched"
    if observation.retrieval_empty or review == "retrieved_evidence_missing":
        return "retrieval_miss"
    if observation.context_empty or review == "context_evidence_missing":
        return "answer_context_missing_fact"
    if observation.generation_outcome == "no_answer" and (
        observation.context_fact_present or review == "context_support_confirmed"
    ):
        return "false_abstention"
    if observation.generation_outcome == "answer" and review == "answer_fact_omitted":
        return "canonical_fact_omission"
    return "semantic_match_uncertain"


def diagnose_case(case, knowledge_base, kb_id, retriever, http_client=requests):
    observation = Observation()
    observed_retriever = ObservedRetriever(retriever, case, observation)
    http = ObservedHttp(http_client, observation)
    try:
        model = ObservedModel(LocalAnswerModel(http_client=http), http, observed_retriever)
        response = AnswerService(retriever=observed_retriever, model=model).answer(
            UUID(kb_id), case.question.strip()
        )
        payload = response.model_dump(mode="json")
        ready_sources = {
            document.filename for document in knowledge_base.documents if document.status == "Ready"
        }
        observation.references_valid = (
            _reference_validity(case, ready_sources, 200, payload) == "valid"
        )
        observation.canonical_match = _canonical_fact_result(case, 200, payload) == "pass"
    except LocalModelError as error:
        if error.category == LocalModelErrorCategory.INVALID_OUTPUT:
            if observation.validation_failure == "none":
                observation.validation_failure = "structured_output"
            observation.generation_outcome = "invalid"
        else:
            observation.execution_status = "dependency_unavailable"
            if observation.generation_invoked:
                observation.generation_outcome = "unavailable"
    except (RetrievalDependencyError, requests.RequestException):
        observation.execution_status = "dependency_unavailable"
    except Exception:
        # No exception text, tracebacks, or raw responses are logged or persisted.
        observation.execution_status = "unexpected_failure"
    return observation


def safe_result(case: EvaluationCase, observation: Observation, review=None) -> dict:
    return {
        "case_id": case.case_id,
        "split": case.split,
        "category": case.category,
        "execution_status": observation.execution_status,
        "classification": classify(observation, review),
        "expected_source_retrieved": observation.expected_source_retrieved,
        "canonical_string_in_retrieved_chunks": observation.retrieved_fact_present,
        "canonical_string_in_model_context": observation.context_fact_present,
        "generation_invoked": observation.generation_invoked,
        "generation_outcome": observation.generation_outcome,
        "validation_failure": observation.validation_failure,
        "review_decision": review,
    }


def run_cases(manifest, knowledge_base_ids, *, retriever=None, diagnose=diagnose_case):
    results = []
    retriever = retriever or SourceRetrievalService()
    for case in manifest.cases:
        if case.expected_outcome != "answer":
            continue
        try:
            observation = diagnose(
                case, manifest.knowledge_bases[case.knowledge_base],
                knowledge_base_ids[case.knowledge_base], retriever,
            )
        except Exception:
            observation = Observation(execution_status="unexpected_failure")
        results.append(safe_result(case, observation))
        print(f"{case.case_id}: {results[-1]['classification'] or 'execution_failure'}", flush=True)
    return results


def validate_results(manifest: EvaluationManifest, results: list[dict]) -> None:
    """Reject unsafe metadata rather than serializing arbitrary caller dictionaries."""
    cases = {case.case_id: case for case in manifest.cases if case.expected_outcome == "answer"}
    if len(cases) != 12 or len(results) != 12 or {r["case_id"] for r in results} != set(cases):
        raise EvaluationError("Diagnostic case coverage is invalid.")
    for result in results:
        case = cases[result["case_id"]]
        expected_keys = set(safe_result(case, Observation()))
        if set(result) != expected_keys:
            raise EvaluationError("Diagnostic metadata is invalid.")
        if result["split"] != case.split or result["category"] != case.category:
            raise EvaluationError("Diagnostic metadata is invalid.")
        if (
            result["execution_status"] not in EXECUTION_STATUSES
            or result["classification"] not in (*CLASSIFICATIONS, None)
            or result["generation_outcome"] not in {
                "answer", "no_answer", "invalid", "not_invoked", "unavailable",
            }
            or result["validation_failure"] not in VALIDATION_CATEGORIES
            or result["review_decision"] not in (*REVIEW_DECISIONS, None)
            or any(type(result[key]) is not bool for key in (
                "expected_source_retrieved", "canonical_string_in_retrieved_chunks",
                "canonical_string_in_model_context", "generation_invoked",
            ))
            or (result["execution_status"] == "ok") != (result["classification"] is not None)
        ):
            raise EvaluationError("Diagnostic metadata is invalid.")


def build_summary(manifest, results):
    validate_results(manifest, results)
    complete = all(result["execution_status"] == "ok" for result in results)
    counts = {key: sum(r["classification"] == key for r in results) for key in CLASSIFICATIONS}
    hypothesis = "resolve_measurement_uncertainty"
    support = [r["case_id"] for r in results if r["classification"] == "semantic_match_uncertain"]
    if complete and not support:
        for category, candidate in (
            ("answer_context_missing_fact", "inspect_answer_evidence_assembly"),
            ("retrieval_miss", "inspect_retrieval_evidence_coverage"),
            ("invalid_reference", "inspect_reference_output_adherence"),
            ("invalid_model_output", "inspect_structured_output_adherence"),
            ("false_abstention", "investigate_confirmed_evidence_abstention"),
            ("canonical_fact_omission", "investigate_confirmed_fact_omission"),
        ):
            support = [r["case_id"] for r in results if r["classification"] == category]
            if support:
                hypothesis = candidate
                break
    if not complete:
        support = [r["case_id"] for r in results if r["execution_status"] != "ok"]
    return {
        "schema_version": 1, "complete": complete, "case_count": len(results),
        "counts": counts,
        "by_split": {
            group: dict(Counter(r["classification"] or "execution_failure" for r in results
                                if r["split"] == group))
            for group in ("calibration", "holdout")
        },
        "by_category": {
            group: dict(Counter(r["classification"] or "execution_failure" for r in results
                                if r["category"] == group))
            for group in ("direct_evidence", "paraphrased_evidence")
        },
        "recommended_hypothesis": hypothesis, "supporting_case_ids": support,
        "results": results,
    }


def write_artifacts(output_directory, manifest, results):
    summary = build_summary(manifest, results)
    lines = [
        "# Grounded-answer diagnostic decision summary", "",
        f"Complete: {summary['complete']}", f"Cases: {summary['case_count']}",
        "v1 is a fixed, previously inspected regression benchmark; it is not unseen validation.",
        "Canonical-string absence is not proof of semantic absence or an incorrect answer.",
        "Human review is unresolved unless a fixed review decision is explicitly recorded.", "",
    ]
    for group in ("counts", "by_split", "by_category"):
        lines.append(f"{group}: {json.dumps(summary[group], sort_keys=True)}")
    lines.extend([
        "", f"One next hypothesis: {summary['recommended_hypothesis']}",
        f"Supporting case IDs: {', '.join(summary['supporting_case_ids']) or 'none'}",
        "Uncertainty: substring checks cannot establish paraphrase equivalence or entailment.",
    ])
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "diagnostic.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "decision-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def check_console(flag, stdin=sys.stdin, stdout=sys.stdout):
    if flag:
        # Even a TTY can be recorded by Docker or a host transcript. No raw mode is enabled.
        if not (stdin.isatty() and stdout.isatty()):
            raise EvaluationError(CONSOLE_MESSAGE)
        raise EvaluationError(CONSOLE_MESSAGE)


def run_from_arguments(arguments):
    check_console(arguments.inspect_console)
    if os.getenv("DIAGNOSTIC_ISOLATED") != "1":
        raise EvaluationError("Run the documented isolated diagnostic command.")
    manifest = load_manifest(arguments.manifest)
    client = RequestsEvaluationClient("http://api:8000")
    try:
        preflight_model_caches(client, arguments.embedding_cache)
    except Exception:
        raise EvaluationError(PREFLIGHT_MESSAGE) from None
    knowledge_base_ids = seed_fixture_stack(
        manifest, fixture_root=arguments.manifest.parent, client=client
    )
    results = run_cases(manifest, knowledge_base_ids)
    summary = write_artifacts(arguments.output_directory, manifest, results)
    print(f"Diagnostic complete: {summary['complete']}", flush=True)
    print(f"One next hypothesis: {summary['recommended_hypothesis']}", flush=True)
    return 0 if summary["complete"] else 2


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose only committed sanitized answer fixtures."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--inspect-console", action="store_true")
    arguments = parser.parse_args()
    # Third-party logs are not diagnostic evidence and must never retain raw content.
    logging.disable(logging.CRITICAL)
    try:
        return run_from_arguments(arguments)
    except EvaluationError as error:
        if str(error) in {PREFLIGHT_MESSAGE, CONSOLE_MESSAGE}:
            print(str(error))
        else:
            print("Diagnostic setup failed safely; check isolated setup and prewarmed caches.")
        return 2
    except Exception:
        print("Diagnostic execution failed safely; no raw details were retained.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
