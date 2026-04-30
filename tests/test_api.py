import pytest
from unittest.mock import MagicMock, patch
import os

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["PINECONE_API_KEY"] = "test-key"
os.environ["VALID_API_KEYS"] = "dev-key"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"


# Mock Pinecone before importing app
with patch("pinecone.Pinecone"), \
     patch("langchain_pinecone.PineconeVectorStore"), \
     patch("app.services.pinecone_service.PineconeService.ensure_index_exists"):
    from fastapi.testclient import TestClient
    from app.main import app

TEST_API_KEY = "dev-key"
HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture()
def client():
    with patch("app.services.pinecone_service.PineconeService.ensure_index_exists"):
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_no_auth(client):
    resp = client.post("/api/v1/query/", json={"question": "hello"})
    assert resp.status_code == 401


def test_query_wrong_key(client):
    resp = client.post(
        "/api/v1/query/",
        json={"question": "hello"},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


@patch("app.api.query.get_rag_service")
def test_query_success(mock_get_rag, client):
    mock_svc = MagicMock()
    mock_svc.query.return_value = {
        "answer": "The answer is 42.",
        "sources": ["https://example.com"],
        "context_docs": [{"content": "Some context", "metadata": {}}],
        "from_cache": False,
    }
    mock_get_rag.return_value = mock_svc

    resp = client.post(
        "/api/v1/query/",
        json={"question": "What is the answer?"},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The answer is 42."
    assert data["question"] == "What is the answer?"


@patch("app.api.ingest.get_pinecone_service")
@patch("app.api.ingest.ingest_urls")
def test_ingest_url(mock_ingest, mock_pinecone, client):
    mock_ingest.return_value = [MagicMock()] * 5
    mock_pinecone.return_value.upsert_documents.return_value = [
        "id1", "id2", "id3", "id4", "id5"
    ]

    resp = client.post(
        "/api/v1/ingest/url",
        json={"urls": ["https://example.com"], "source_type": "web"},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_ingested"] == 5