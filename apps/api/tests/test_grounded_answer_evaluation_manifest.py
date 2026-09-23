"""Validation for committed grounded-answer evaluation fixtures.

The manifest is intentionally validated with only Python standard-library modules.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "grounded_answers"
VERSION_ROOT = FIXTURE_ROOT / "v1"
MANIFEST_PATH = VERSION_ROOT / "manifest.json"
CASE_ID_PATTERN = re.compile(r"^[a-z]+(?:-[a-z]+)*-[0-9]{3}$")
EXPECTED_CATEGORY_COUNTS = {
    "direct_evidence": 8,
    "paraphrased_evidence": 4,
    "no_supporting_evidence": 4,
    "knowledge_base_isolation": 2,
    "non_ready_unavailable": 2,
}
EXPECTED_SPLIT_COUNTS = {"calibration": 10, "holdout": 10}


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def document_index(manifest: dict[str, object]) -> dict[tuple[str, str], dict[str, str]]:
    documents: dict[tuple[str, str], dict[str, str]] = {}
    for knowledge_base in manifest["knowledge_bases"]:
        knowledge_base_id = knowledge_base["id"]
        for document in knowledge_base["documents"]:
            key = (knowledge_base_id, document["filename"])
            assert key not in documents, f"Duplicate fixture document: {key}"
            documents[key] = document
    return documents


def document_text(document: dict[str, str]) -> str:
    fixture_path = (VERSION_ROOT / document["path"]).resolve()
    assert fixture_path.is_relative_to(VERSION_ROOT.resolve())
    assert fixture_path.is_file()
    return fixture_path.read_text(encoding="utf-8")


def test_manifest_loads_with_standard_library_only() -> None:
    code = (
        "import json, pathlib, sys; "
        "manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')); "
        "assert manifest['schema_version'] == 1; "
        "assert len(manifest['cases']) == 20"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(MANIFEST_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_manifest_schema_version_and_case_ids_are_stable() -> None:
    manifest = load_manifest()
    cases = manifest["cases"]
    case_ids = [case["id"] for case in cases]

    assert manifest["schema_version"] == 1
    assert len(case_ids) == len(set(case_ids))
    assert all(CASE_ID_PATTERN.fullmatch(case_id) for case_id in case_ids)


def test_manifest_has_exact_category_and_split_counts() -> None:
    manifest = load_manifest()
    cases = manifest["cases"]

    assert len(cases) == 20
    assert Counter(case["category"] for case in cases) == EXPECTED_CATEGORY_COUNTS
    assert Counter(case["split"] for case in cases) == EXPECTED_SPLIT_COUNTS


def test_fixture_documents_exist_and_knowledge_base_references_are_valid() -> None:
    manifest = load_manifest()
    knowledge_base_ids = {knowledge_base["id"] for knowledge_base in manifest["knowledge_bases"]}
    documents = document_index(manifest)

    assert len(knowledge_base_ids) == len(manifest["knowledge_bases"])
    assert all(knowledge_base_id.isascii() for knowledge_base_id in knowledge_base_ids)

    for document in documents.values():
        assert document["path"].endswith(".txt")
        assert document_text(document).strip()

    for case in manifest["cases"]:
        assert case["knowledge_base"] in knowledge_base_ids


def test_answer_cases_require_ready_selected_evidence_and_expected_fields() -> None:
    manifest = load_manifest()
    documents = document_index(manifest)

    for case in manifest["cases"]:
        if case["expected_outcome"] == "answer":
            source_filename = case.get("expected_source_filename")
            canonical_fact = case.get("expected_canonical_fact")

            assert isinstance(source_filename, str) and source_filename
            assert isinstance(canonical_fact, str) and canonical_fact
            document = documents[(case["knowledge_base"], source_filename)]
            assert document["status"] == "Ready"
            assert canonical_fact in document_text(document)
        else:
            assert case["expected_outcome"] == "no_answer"
            assert "expected_source_filename" not in case
            assert "expected_canonical_fact" not in case


def test_isolation_cases_identify_matching_evidence_outside_selected_knowledge_base() -> None:
    manifest = load_manifest()
    documents = document_index(manifest)
    isolation_cases = [
        case for case in manifest["cases"] if case["category"] == "knowledge_base_isolation"
    ]

    assert len(isolation_cases) == 2
    for case in isolation_cases:
        metadata = case["outside_matching_evidence"]
        assert case["expected_outcome"] == "no_answer"
        assert metadata["knowledge_base"] != case["knowledge_base"]
        document = documents[(metadata["knowledge_base"], metadata["source_filename"])]
        assert document["status"] == "Ready"
        assert metadata["canonical_fact"] in document_text(document)


def test_non_ready_cases_identify_matching_unavailable_fixture_document() -> None:
    manifest = load_manifest()
    documents = document_index(manifest)
    unavailable_cases = [
        case for case in manifest["cases"] if case["category"] == "non_ready_unavailable"
    ]

    assert len(unavailable_cases) == 2
    for case in unavailable_cases:
        metadata = case["unavailable_matching_evidence"]
        assert case["expected_outcome"] == "no_answer"
        assert metadata["knowledge_base"] == case["knowledge_base"]
        assert metadata["status"] in {"Uploaded", "Failed"}
        document = documents[(metadata["knowledge_base"], metadata["source_filename"])]
        assert document["status"] == metadata["status"]
        assert metadata["canonical_fact"] in document_text(document)


def test_fixtures_are_plain_text_and_exclude_common_secret_markers() -> None:
    manifest = load_manifest()
    prohibited_markers = ("BEGIN PRIVATE KEY", "AKIA", "sk-", "password=")

    for document in document_index(manifest).values():
        content = document_text(document)
        assert not any(marker in content for marker in prohibited_markers)
