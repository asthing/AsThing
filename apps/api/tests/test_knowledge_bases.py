from datetime import datetime

import pytest

from app.database import SessionLocal
from app.knowledge_bases import KnowledgeBase


def test_create_knowledge_base_trims_name_and_returns_utc_timestamp(client) -> None:
    response = client.post("/knowledge-bases", json={"name": "  Engineering Notes  "})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Engineering Notes"
    assert datetime.fromisoformat(body["created_at"]).tzinfo is not None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"name": None},
        {"name": ""},
        {"name": "   "},
        {"name": "a" * 101},
    ],
)
def test_create_knowledge_base_rejects_invalid_names(client, payload) -> None:
    response = client.post("/knowledge-bases", json=payload)

    assert response.status_code == 422


def test_create_knowledge_base_rejects_case_insensitive_duplicate(client) -> None:
    assert client.post("/knowledge-bases", json={"name": "Engineering Notes"}).status_code == 201

    response = client.post("/knowledge-bases", json={"name": " engineering notes "})

    assert response.status_code == 409
    assert response.json() == {
        "detail": 'A knowledge base named "engineering notes" already exists.'
    }
    assert len(client.get("/knowledge-bases").json()) == 1


def test_list_knowledge_bases_returns_newest_first(client) -> None:
    session = SessionLocal()
    try:
        older = KnowledgeBase(
            name="Older",
            normalized_name="older",
            created_at=datetime.fromisoformat("2026-09-12T08:00:00+00:00"),
        )
        newer = KnowledgeBase(
            name="Newer",
            normalized_name="newer",
            created_at=datetime.fromisoformat("2026-09-12T09:00:00+00:00"),
        )
        session.add_all([older, newer])
        session.commit()
    finally:
        session.close()

    response = client.get("/knowledge-bases")

    assert response.status_code == 200
    assert [knowledge_base["name"] for knowledge_base in response.json()] == ["Newer", "Older"]
