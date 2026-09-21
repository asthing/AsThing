import json
from uuid import UUID, uuid4

import pytest
import requests

from app.answers import AnswerService, get_answer_service
from app.database import SessionLocal
from app.document_models import Document, DocumentChunk, DocumentStatus
from app.embeddings import EMBEDDING_DIMENSION, EmbeddingBatch, EmbeddingModelMetadata
from app.knowledge_bases import KnowledgeBase
from app.local_llm import (
    OLLAMA_MODEL,
    OLLAMA_MODEL_DIGEST,
    SYSTEM_INSTRUCTION,
    LocalAnswerModel,
    LocalModelError,
    LocalModelErrorCategory,
    ModelDecision,
    PromptSource,
)
from app.main import app
from app.retrieval import SourceRetrievalService
from app.vector_store import VectorSearchHit, VectorStoreError, VectorStoreErrorCategory


class Embeddings:
    def embed(self, texts):
        return EmbeddingBatch(
            vectors=[[0.1] * EMBEDDING_DIMENSION for _ in texts],
            model=EmbeddingModelMetadata("BAAI/bge-m3", "pinned-revision", EMBEDDING_DIMENSION),
        )


class Vectors:
    def __init__(self, hits=(), error=None):
        self.hits = list(hits)
        self.error = error

    def search(self, _vector, **_options):
        if self.error:
            raise self.error
        return self.hits


class Model:
    def __init__(self, decision=None, error=None):
        self.decision = decision
        self.error = error
        self.calls = []

    def generate(self, question, sources):
        self.calls.append((question, sources))
        if self.error:
            raise self.error
        return self.decision


def knowledge_base(name="Selected"):
    session = SessionLocal()
    try:
        kb = KnowledgeBase(name=name, normalized_name=name.casefold())
        session.add(kb)
        session.commit()
        session.refresh(kb)
        return kb.id
    finally:
        session.close()


def chunk(kb_id: UUID, filename: str, content: str, status=DocumentStatus.READY):
    session = SessionLocal()
    try:
        document = Document(
            knowledge_base_id=kb_id,
            original_filename=filename,
            file_size_bytes=100,
            file_type="txt",
            status=status,
            storage_object_key=f"documents/{uuid4()}",
        )
        session.add(document)
        session.flush()
        item = DocumentChunk(
            document_id=document.id,
            ordinal=0,
            content=content,
            content_hash=uuid4().hex,
            qdrant_point_id=uuid4(),
        )
        session.add(item)
        session.commit()
        session.refresh(item)
        return item.id
    finally:
        session.close()


def service(hits=(), decision=None, error=None):
    return AnswerService(
        retriever=SourceRetrievalService(
            embedding_service=Embeddings(),
            vector_store=Vectors(hits),
        ),
        model=Model(decision, error),
    )


def test_answer_endpoint_rejects_blank_and_unknown_kb(client):
    kb_id = knowledge_base()
    blank = client.post(f"/knowledge-bases/{kb_id}/answers", json={"question": "   "})
    missing = client.post(f"/knowledge-bases/{uuid4()}/answers", json={"question": "Why?"})
    assert blank.status_code == 422
    assert "Enter a question" in str(blank.json())
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Knowledge base not found."}


def test_no_ready_and_no_relevant_evidence_return_no_answer_without_model_call():
    kb_id = knowledge_base()
    chunk(kb_id, "pending.txt", "not ready", DocumentStatus.PROCESSING)
    answer_service = service()
    no_ready = answer_service.answer(kb_id, "Question?")
    chunk(kb_id, "ready.txt", "some source")
    no_results = answer_service.answer(kb_id, "Question?")
    assert no_ready.outcome == no_results.outcome == "no_answer"
    assert no_ready.reason == "no_ready_documents"
    assert no_results.reason == "insufficient_evidence"
    assert no_ready.references == no_results.references == []
    assert answer_service.model.calls == []


def test_grounded_answer_uses_only_selected_ready_postgres_chunks_and_public_references(client):
    selected = knowledge_base()
    other = knowledge_base("Other")
    ready = chunk(selected, "policy.txt", "Annual leave is requested in the HR portal.")
    stale = chunk(selected, "draft.txt", "Do not use this.", DocumentStatus.UPLOADED)
    foreign = chunk(other, "private.txt", "Other KB secret.")
    answer_service = service(
        [
            VectorSearchHit(foreign, 0.99),
            VectorSearchHit(stale, 0.95),
            VectorSearchHit(ready, 0.9),
        ],
        ModelDecision(outcome="answer", answer="Use the HR portal.", references=["source-1"]),
    )
    app.dependency_overrides[get_answer_service] = lambda: answer_service
    try:
        response = client.post(
            f"/knowledge-bases/{selected}/answers",
            json={"question": "  How is leave requested?  "},
        )
    finally:
        app.dependency_overrides.pop(get_answer_service, None)
    assert response.status_code == 200
    assert response.json() == {
        "outcome": "answer",
        "answer": "Use the HR portal.",
        "reason": None,
        "message": None,
        "references": [
            {
                "reference_key": "source-1",
                "source_filename": "policy.txt",
                "file_type": "txt",
                "excerpt": "Annual leave is requested in the HR portal.",
            }
        ],
    }
    question, sources = answer_service.model.calls[0]
    assert question == "How is leave requested?"
    assert [(source.key, source.content) for source in sources] == [
        ("source-1", "Annual leave is requested in the HR portal.")
    ]
    assert str(ready) not in str(response.json())
    assert str(foreign) not in str(response.json())


def test_explicit_model_no_answer_and_unknown_reference():
    kb_id = knowledge_base()
    ready = chunk(kb_id, "policy.txt", "Leave is requested in the HR portal.")
    hits = [VectorSearchHit(ready, 0.9)]
    no_answer = service(
        hits,
        ModelDecision(outcome="no_answer", answer=None, references=[]),
    ).answer(kb_id, "How much leave?")
    assert no_answer.outcome == "no_answer"
    assert no_answer.reason == "insufficient_evidence"
    assert no_answer.references == []

    unknown = service(
        hits,
        ModelDecision(outcome="answer", answer="Invented.", references=["source-999"]),
    )
    with pytest.raises(LocalModelError) as error:
        unknown.answer(kb_id, "How much leave?")
    assert error.value.category == LocalModelErrorCategory.INVALID_OUTPUT


def test_retrieval_and_model_failures_are_safe_503(client):
    kb_id = knowledge_base()
    ready = chunk(kb_id, "policy.txt", "Leave is requested in the HR portal.")
    failures = [
        AnswerService(
            retriever=SourceRetrievalService(
                embedding_service=Embeddings(),
                vector_store=Vectors(error=VectorStoreError(VectorStoreErrorCategory.QUERY_FAILED)),
            ),
            model=Model(),
        ),
        service(
            [VectorSearchHit(ready, 0.9)],
            error=LocalModelError(LocalModelErrorCategory.UNAVAILABLE),
        ),
    ]
    for answer_service in failures:
        app.dependency_overrides[get_answer_service] = lambda: answer_service
        try:
            response = client.post(
                f"/knowledge-bases/{kb_id}/answers", json={"question": "What is the policy?"}
            )
        finally:
            app.dependency_overrides.pop(get_answer_service, None)
        assert response.status_code == 503
        assert response.json() == {
            "detail": "Answer generation is temporarily unavailable. Please try again."
        }


class HttpResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class HttpClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def get(self, url, *, timeout):
        self.calls.append(("get", url, timeout))
        model = {"name": OLLAMA_MODEL, "digest": OLLAMA_MODEL_DIGEST.removeprefix("sha256:")}
        return HttpResponse({"models": [model]})

    def post(self, url, *, json, timeout):
        self.calls.append(("post", url, json, timeout))
        if isinstance(self.output, Exception):
            raise self.output
        return HttpResponse({"message": {"content": self.output}})


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps({"outcome": "answer", "answer": "Unsupported", "references": []}),
        json.dumps({"outcome": "answer", "answer": "x" * 701, "references": ["source-1"]}),
        json.dumps({"outcome": "no_answer", "answer": "invented", "references": []}),
    ],
)
def test_local_model_rejects_malformed_or_unsafe_output(content):
    model = LocalAnswerModel(http_client=HttpClient(content))
    with pytest.raises(LocalModelError) as error:
        model.generate("Question?", [PromptSource("source-1", "Source text")])
    assert error.value.category == LocalModelErrorCategory.INVALID_OUTPUT


def test_local_model_checks_digest_and_sends_only_question_and_evidence():
    http = HttpClient(json.dumps({
        "outcome": "answer", "answer": "Use the portal.", "references": ["source-1"]
    }))
    result = LocalAnswerModel(http_client=http).generate(
        "How?", [PromptSource("source-1", "Use the portal.")]
    )
    assert result.answer == "Use the portal."
    assert http.calls[0][1].endswith("/api/tags")
    payload = http.calls[1][2]
    assert payload["model"] == OLLAMA_MODEL
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert json.loads(payload["messages"][1]["content"]) == {
        "question": "How?", "sources": [{"key": "source-1", "text": "Use the portal."}]
    }
    assert len(http.calls) == 2


def test_system_instruction_requires_one_complete_valid_output_shape():
    answer_shape = '"outcome":"answer","answer":"a concise answer supported by the sources"'
    assert answer_shape in SYSTEM_INSTRUCTION
    assert '"outcome":"no_answer","answer":null,"references":[]' in SYSTEM_INSTRUCTION
    assert 'Never return outcome "answer" with a null or empty answer' in SYSTEM_INSTRUCTION


def test_unavailable_or_timed_out_ollama_has_safe_error():
    for output in (requests.Timeout("secret internal URL"), requests.ConnectionError("secret")):
        model = LocalAnswerModel(http_client=HttpClient(output))
        with pytest.raises(LocalModelError) as error:
            model.generate("Question?", [PromptSource("source-1", "Source text")])
        assert "secret" not in str(error.value)


def test_model_digest_mismatch_prevents_generation():
    class WrongDigest(HttpClient):
        def get(self, url, *, timeout):
            return HttpResponse({"models": [{"name": OLLAMA_MODEL, "digest": "sha256:wrong"}]})

    http = WrongDigest("{}")
    with pytest.raises(LocalModelError) as error:
        LocalAnswerModel(http_client=http).generate(
            "Question?", [PromptSource("source-1", "Source text")]
        )
    assert error.value.category == LocalModelErrorCategory.UNAVAILABLE
    assert http.calls == []


def test_local_model_allows_only_one_active_generation():
    model = LocalAnswerModel(http_client=HttpClient("{}"))
    assert model._generation_slot.acquire(blocking=False)
    try:
        with pytest.raises(LocalModelError) as error:
            model.generate("Question?", [PromptSource("source-1", "Source text")])
        assert error.value.category == LocalModelErrorCategory.BUSY
    finally:
        model._generation_slot.release()
