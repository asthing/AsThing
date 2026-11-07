import json
import logging
from argparse import Namespace
from dataclasses import replace
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pytest
import requests

from app import grounded_answer_diagnostic as diagnostic
from app.grounded_answer_evaluation import EvaluationError, load_manifest
from app.local_llm import OLLAMA_MODEL, OLLAMA_MODEL_DIGEST
from app.retrieval import EvidenceSearchResult, RetrievalOutcome, RetrievedChunk

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "grounded_answers" / "v1"
SENTINEL = "PRIVATE_SENTINEL http://secret.example credential prompt source-answer"


def manifest():
    return load_manifest(FIXTURE_ROOT / "manifest.json")


def case():
    return next(c for c in manifest().cases if c.expected_outcome == "answer")


@pytest.mark.parametrize(
    ("changes", "review", "expected"),
    [
        ({"canonical_match": True, "references_valid": True}, None, "matched"),
        ({"retrieval_empty": True}, None, "retrieval_miss"),
        ({"retrieved_fact_present": True, "context_empty": True}, None,
         "answer_context_missing_fact"),
        ({"generation_outcome": "no_answer", "context_fact_present": True}, None,
         "false_abstention"),
        ({"validation_failure": "reference", "retrieval_empty": True}, None,
         "invalid_reference"),
        ({"validation_failure": "structured_output", "retrieval_empty": True}, None,
         "invalid_model_output"),
        ({"generation_outcome": "answer"}, None, "semantic_match_uncertain"),
        ({"generation_outcome": "answer"}, "answer_fact_omitted", "canonical_fact_omission"),
        ({"generation_outcome": "no_answer"}, None, "semantic_match_uncertain"),
        ({"retrieved_fact_present": True, "context_fact_present": False}, None,
         "semantic_match_uncertain"),
        ({}, "retrieved_evidence_missing", "retrieval_miss"),
        ({}, "context_evidence_missing", "answer_context_missing_fact"),
        ({"generation_outcome": "no_answer"}, "context_support_confirmed", "false_abstention"),
        ({"execution_status": "dependency_unavailable", "validation_failure": "reference"},
         None, None),
    ],
)
def test_classification_precedence_and_uncertainty(changes, review, expected):
    assert diagnostic.classify(diagnostic.Observation(**changes), review) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (SENTINEL, "structured_output"),
        ({"outcome": "invalid"}, "structured_output"),
        ({"outcome": "answer", "answer": "fact"}, "reference"),
        ({"outcome": "answer", "answer": "fact", "references": []}, "reference"),
        ({"outcome": "answer", "answer": "fact", "references": [None]}, "reference"),
        ({"outcome": "answer", "answer": "fact", "references": ["source-9"]}, "reference"),
        ({"outcome": "answer", "answer": "fact", "references": ["source-1"] * 2}, "reference"),
        ({"outcome": "no_answer", "answer": None, "references": ["source-1"]}, "reference"),
        ({"outcome": "answer", "answer": "x" * 701, "references": ["source-1"]},
         "structured_output"),
        ({"outcome": "answer", "answer": "fact", "references": ["source-1"]}, "none"),
        ({"outcome": "no_answer", "answer": None, "references": []}, "none"),
    ],
)
def test_specific_model_validation_categories(payload, expected):
    content = json.dumps(payload) if isinstance(payload, dict) else payload
    assert diagnostic.output_failure(content, {"source-1"}) == expected


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class Http:
    def __init__(self, content):
        self.content = content
        self.generations = 0
        self.payload = None

    def get(self, *_args, **_kwargs):
        return Response({"models": [{"name": OLLAMA_MODEL, "digest": OLLAMA_MODEL_DIGEST}]})

    def post(self, *_args, **kwargs):
        self.generations += 1
        self.payload = kwargs["json"]
        if isinstance(self.content, Exception):
            raise self.content
        return Response({"message": {"content": self.content}})


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def search_evidence(self, *_args):
        self.calls += 1
        return EvidenceSearchResult(
            RetrievalOutcome.RESULTS if self.chunks else RetrievalOutcome.NO_RESULTS, self.chunks
        )


def evidence(content, filename=None):
    return RetrievedChunk(uuid4(), content, filename or case().expected_source_filename, "txt")


def diagnose(content, chunks=None):
    selected = case()
    http = Http(content)
    retriever = Retriever(
        chunks if chunks is not None else [evidence(selected.expected_canonical_fact)]
    )
    observation = diagnostic.diagnose_case(
        selected, manifest().knowledge_bases[selected.knowledge_base], str(uuid4()), retriever, http
    )
    return observation, http, retriever


def test_observes_real_answer_service_context_and_single_generation():
    selected = case()
    content = json.dumps({
        "outcome": "answer", "answer": selected.expected_canonical_fact,
        "references": ["source-1"],
    })
    observation, http, retriever = diagnose(content)
    assert diagnostic.classify(observation) == "matched"
    assert observation.expected_source_retrieved
    assert observation.retrieved_fact_present and observation.context_fact_present
    assert observation.generation_invoked
    assert observation.generation_outcome == "answer"
    assert http.generations == retriever.calls == 1


def test_actual_top_one_context_does_not_get_replaced_with_expected_fact():
    chunks = [evidence("different top chunk"), evidence(case().expected_canonical_fact)]
    observation, http, _ = diagnose(
        json.dumps({"outcome": "no_answer", "answer": None, "references": []}), chunks
    )
    assert observation.retrieved_fact_present
    assert not observation.context_fact_present
    supplied = json.loads(http.payload["messages"][1]["content"])["sources"]
    assert supplied == [{"key": "source-1", "text": "different top chunk"}]
    assert diagnostic.classify(observation) == "semantic_match_uncertain"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("not-json", "invalid_model_output"),
        (json.dumps({"outcome": "answer", "answer": "fact", "references": ["unknown"]}),
         "invalid_reference"),
        (json.dumps({"outcome": "no_answer", "answer": None, "references": []}),
         "false_abstention"),
        (requests.Timeout(SENTINEL), None),
        (requests.ConnectionError(SENTINEL), None),
    ],
)
def test_production_component_errors_are_observed_without_second_generation(content, expected):
    observation, http, _ = diagnose(content)
    assert diagnostic.classify(observation) == expected
    assert http.generations == 1
    if expected is None:
        assert observation.execution_status == "dependency_unavailable"
        assert observation.generation_invoked
        assert observation.generation_outcome == "unavailable"


def test_no_evidence_does_not_invoke_generation():
    observation, http, _ = diagnose("unused", [])
    assert diagnostic.classify(observation) == "retrieval_miss"
    assert not observation.generation_invoked
    assert http.generations == 0


def test_runner_continues_after_execution_failure_and_only_runs_expected_answers(capsys):
    selected_manifest = manifest()
    calls = []

    def fake(case, *_args):
        calls.append(case.case_id)
        if len(calls) == 1:
            raise RuntimeError(SENTINEL)
        return diagnostic.Observation(generation_outcome="no_answer", context_fact_present=True)

    results = diagnostic.run_cases(
        selected_manifest, {key: str(uuid4()) for key in selected_manifest.knowledge_bases},
        retriever=Retriever([]), diagnose=fake,
    )
    assert len(results) == len(set(calls)) == 12
    assert results[0]["classification"] is None
    summary = diagnostic.build_summary(selected_manifest, results)
    assert not summary["complete"]
    assert summary["case_count"] == 12
    assert SENTINEL not in capsys.readouterr().out


def safe_results():
    return [diagnostic.safe_result(c, diagnostic.Observation())
            for c in manifest().cases if c.expected_outcome == "answer"]


def test_complete_artifacts_are_safe_and_grouped_by_split_and_category(tmp_path, caplog):
    results = safe_results()
    summary = diagnostic.write_artifacts(tmp_path, manifest(), results)
    assert summary["complete"]
    assert summary["case_count"] == 12
    assert summary["counts"]["semantic_match_uncertain"] == 12
    assert summary["by_split"] == {
        "calibration": {"semantic_match_uncertain": 6},
        "holdout": {"semantic_match_uncertain": 6},
    }
    assert summary["by_category"] == {
        "direct_evidence": {"semantic_match_uncertain": 8},
        "paraphrased_evidence": {"semantic_match_uncertain": 4},
    }
    assert summary["recommended_hypothesis"] == "resolve_measurement_uncertainty"
    assert len(summary["supporting_case_ids"]) == 12
    persisted = "".join(path.read_text() for path in tmp_path.iterdir())
    for c in manifest().cases:
        assert c.question not in persisted
        if c.expected_canonical_fact:
            assert c.expected_canonical_fact not in persisted
    assert SENTINEL not in persisted + caplog.text
    assert "previously inspected" in persisted


def test_privacy_sentinels_do_not_escape_observers_reports_or_logs(tmp_path, caplog, capsys):
    raw = json.dumps({"outcome": "answer", "answer": SENTINEL, "references": ["source-1"]})
    with caplog.at_level(logging.DEBUG):
        observation, _, _ = diagnose(raw, [evidence(SENTINEL)])
    results = safe_results()
    results[0] = diagnostic.safe_result(case(), observation)
    diagnostic.write_artifacts(tmp_path, manifest(), results)
    serialized = "".join(path.read_text() for path in tmp_path.iterdir())
    captured = capsys.readouterr()
    assert SENTINEL not in serialized + caplog.text + captured.out + captured.err
    assert "http://" not in serialized
    assert raw not in serialized


@pytest.mark.parametrize("field", ["extra", "generation_outcome", "classification", "case_id"])
def test_writer_rejects_arbitrary_content_and_ids(tmp_path, field):
    results = safe_results()
    results[0][field] = SENTINEL
    with pytest.raises((EvaluationError, KeyError)):
        diagnostic.write_artifacts(tmp_path, manifest(), results)
    assert list(tmp_path.iterdir()) == []


def test_exact_coverage_required_and_duplicates_rejected():
    results = safe_results()
    with pytest.raises(EvaluationError):
        diagnostic.build_summary(manifest(), results[:-1])
    results[-1] = results[0]
    with pytest.raises(EvaluationError):
        diagnostic.build_summary(manifest(), results)


@pytest.mark.parametrize("interactive", [False, True])
def test_console_opt_in_never_enables_unrecorded_path_without_guarantee(interactive):
    class Terminal(StringIO):
        def isatty(self):
            return interactive

    output = Terminal()
    diagnostic.check_console(False, Terminal(), output)
    with pytest.raises(EvaluationError, match="Raw console inspection is disabled"):
        diagnostic.check_console(True, Terminal(), output)
    assert output.getvalue() == ""


def test_global_preflight_failure_stops_before_seed_or_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("DIAGNOSTIC_ISOLATED", "1")

    def unavailable(*_args):
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(diagnostic, "preflight_model_caches", unavailable)
    monkeypatch.setattr(
        diagnostic, "seed_fixture_stack", lambda *_a, **_k: pytest.fail("must not seed")
    )
    monkeypatch.setattr(diagnostic, "run_cases", lambda *_a, **_k: pytest.fail("must not run"))
    with pytest.raises(EvaluationError, match=diagnostic.PREFLIGHT_MESSAGE):
        diagnostic.run_from_arguments(Namespace(
            inspect_console=False, manifest=FIXTURE_ROOT / "manifest.json",
            output_directory=tmp_path, embedding_cache=tmp_path,
        ))
    assert list(tmp_path.iterdir()) == []


def test_no_source_or_prompt_changes_and_fixtures_unchanged():
    before = {path: path.read_bytes() for path in FIXTURE_ROOT.rglob("*") if path.is_file()}
    selected = case()
    http = Http(json.dumps({"outcome": "no_answer", "answer": None, "references": []}))
    # Compare observed execution to the unchanged model payload, including the system instruction.
    from app.local_llm import SYSTEM_INSTRUCTION

    diagnostic.diagnose_case(
        selected, manifest().knowledge_bases[selected.knowledge_base], str(uuid4()),
        Retriever([evidence(selected.expected_canonical_fact)]), http,
    )
    assert http.payload["messages"][0]["content"] == SYSTEM_INSTRUCTION
    assert all(path.read_bytes() == content for path, content in before.items())


def test_review_category_is_fixed_and_does_not_override_validation():
    result = diagnostic.safe_result(
        case(), diagnostic.Observation(generation_outcome="answer"), "answer_fact_omitted"
    )
    assert result["review_decision"] == "answer_fact_omitted"
    assert result["classification"] == "canonical_fact_omission"
    assert diagnostic.classify(
        diagnostic.Observation(validation_failure="reference"), "answer_fact_omitted"
    ) == "invalid_reference"


def test_missing_fact_substring_is_not_semantic_miss():
    selected = replace(case(), expected_canonical_fact="retained for 30 days")
    assert not diagnostic.contains_fact(selected, "Keep logs for thirty days.")
    assert diagnostic.classify(diagnostic.Observation()) == "semantic_match_uncertain"
